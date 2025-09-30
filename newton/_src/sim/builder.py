# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for building Newton models."""

from __future__ import annotations

import copy
import ctypes
import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp

from ..core.types import (
    Axis,
    AxisType,
    Devicelike,
    Mat33,
    Quat,
    Sequence,
    Transform,
    Vec3,
    Vec4,
    axis_to_vec3,
    flag_to_int,
    nparray,
)
from ..geometry import (
    MESH_MAXHULLVERT,
    SDF,
    GeoType,
    Mesh,
    ParticleFlags,
    ShapeFlags,
    compute_shape_inertia,
    compute_shape_radius,
    transform_inertia,
)
from ..geometry.inertia import validate_and_correct_inertia_kernel, verify_and_correct_inertia
from ..geometry.utils import RemeshingMethod, compute_obb, remesh_mesh
from .graph_coloring import ColoringAlgorithm, color_trimesh, combine_independent_particle_coloring
from .joints import (
    EqType,
    JointMode,
    JointType,
    get_joint_dof_count,
)
from .model import Model


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    represents the scene using standard Python data structures like lists,
    which are convenient but unsuitable for efficient simulation.
    Call :meth:`finalize` to construct a simulation-ready Model.

    Example
    -------

    .. testcode::

        import newton
        from newton.solvers import SolverXPBD

        builder = newton.ModelBuilder()

        # anchor point (zero mass)
        builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        # build chain
        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

        # create model
        model = builder.finalize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = SolverXPBD(model)

        for i in range(10):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

    Environment Grouping
    --------------------

    ModelBuilder supports environment grouping to organize entities for multi-environment simulations.
    Each entity (particle, body, shape, joint, articulation) has an associated group index:

    - Group -1: Global entities shared across all environments (e.g., ground plane)
    - Group 0, 1, 2, ...: Environment-specific entities

    There are two ways to assign environment groups:

    1. **Direct entity creation**: Entities inherit the builder's `current_env_group` value::

           builder = ModelBuilder()
           builder.current_env_group = -1  # Following entities will be global
           builder.add_ground_plane()
           builder.current_env_group = 0  # Following entities will be in environment 0
           builder.add_body(...)

    2. **Using add_builder()**: ALL entities from the sub-builder are assigned to the specified group::

           robot = ModelBuilder()
           robot.add_body(...)  # Group assignments here will be overridden

           main = ModelBuilder()
           main.add_builder(robot, environment=0)  # All robot entities -> group 0
           main.add_builder(robot, environment=1)  # All robot entities -> group 1

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    @dataclass
    class ShapeConfig:
        """
        Represents the properties of a collision shape used in simulation.
        """

        density: float = 1000.0
        """The density of the shape material."""
        ke: float = 1.0e5
        """The contact elastic stiffness."""
        kd: float = 1000.0
        """The contact damping stiffness."""
        kf: float = 1000.0
        """The contact friction stiffness."""
        ka: float = 0.0
        """The contact adhesion distance."""
        mu: float = 0.5
        """The coefficient of friction."""
        restitution: float = 0.0
        """The coefficient of restitution."""
        thickness: float = 1e-5
        """The thickness of the shape."""
        is_solid: bool = True
        """Indicates whether the shape is solid or hollow. Defaults to True."""
        collision_group: int = -1
        """The collision group ID for the shape. Defaults to -1."""
        collision_filter_parent: bool = True
        """Whether to inherit collision filtering from the parent. Defaults to True."""
        has_shape_collision: bool = True
        """Whether the shape can collide with other shapes. Defaults to True."""
        has_particle_collision: bool = True
        """Whether the shape can collide with particles. Defaults to True."""
        is_visible: bool = True
        """Indicates whether the shape is visible in the simulation. Defaults to True."""

        @property
        def flags(self) -> int:
            """Returns the flags for the shape."""

            shape_flags = ShapeFlags.VISIBLE if self.is_visible else 0
            shape_flags |= ShapeFlags.COLLIDE_SHAPES if self.has_shape_collision else 0
            shape_flags |= ShapeFlags.COLLIDE_PARTICLES if self.has_particle_collision else 0
            return shape_flags

        @flags.setter
        def flags(self, value: int):
            """Sets the flags for the shape."""

            self.is_visible = bool(value & ShapeFlags.VISIBLE)
            self.has_shape_collision = bool(value & ShapeFlags.COLLIDE_SHAPES)
            self.has_particle_collision = bool(value & ShapeFlags.COLLIDE_PARTICLES)

        def copy(self) -> ModelBuilder.ShapeConfig:
            return copy.copy(self)

    class JointDofConfig:
        """
        Describes a joint axis (a single degree of freedom) that can have limits and be driven towards a target.
        """

        def __init__(
            self,
            axis: AxisType | Vec3 = Axis.X,
            limit_lower: float = -1e6,
            limit_upper: float = 1e6,
            limit_ke: float = 1e4,
            limit_kd: float = 1e1,
            target: float = 0.0,
            target_ke: float = 0.0,
            target_kd: float = 0.0,
            mode: int = JointMode.TARGET_POSITION,
            armature: float = 1e-2,
            effort_limit: float = 1e6,
            velocity_limit: float = 1e6,
            friction: float = 0.0,
        ):
            self.axis = wp.normalize(axis_to_vec3(axis))
            """The 3D axis that this JointDofConfig object describes."""
            self.limit_lower = limit_lower
            """The lower position limit of the joint axis. Defaults to -1e6."""
            self.limit_upper = limit_upper
            """The upper position limit of the joint axis. Defaults to 1e6."""
            self.limit_ke = limit_ke
            """The elastic stiffness of the joint axis limits. Defaults to 1e4."""
            self.limit_kd = limit_kd
            """The damping stiffness of the joint axis limits. Defaults to 1e1."""
            self.target = target
            """The target position or velocity (depending on the mode) of the joint axis.
            If `mode` is `JointMode.TARGET_POSITION` and the initial `target` is outside the limits,
            it defaults to the midpoint of `limit_lower` and `limit_upper`. Otherwise, defaults to 0.0."""
            self.target_ke = target_ke
            """The proportional gain of the target drive PD controller. Defaults to 0.0."""
            self.target_kd = target_kd
            """The derivative gain of the target drive PD controller. Defaults to 0.0."""
            self.mode = mode
            """The mode of the joint axis (e.g., `JointMode.TARGET_POSITION` or `JointMode.TARGET_VELOCITY`). Defaults to `JointMode.TARGET_POSITION`."""
            self.armature = armature
            """Artificial inertia added around the joint axis. Defaults to 1e-2."""
            self.effort_limit = effort_limit
            """Maximum effort (force or torque) the joint axis can exert. Defaults to 1e6."""
            self.velocity_limit = velocity_limit
            """Maximum velocity the joint axis can achieve. Defaults to 1e6."""
            self.friction = friction
            """Friction coefficient for the joint axis. Defaults to 0.0."""

            if self.mode == JointMode.TARGET_POSITION and (
                self.target > self.limit_upper or self.target < self.limit_lower
            ):
                self.target = 0.5 * (self.limit_lower + self.limit_upper)

        @classmethod
        def create_unlimited(cls, axis: AxisType | Vec3) -> ModelBuilder.JointDofConfig:
            """Creates a JointDofConfig with no limits."""
            return ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=-1e6,
                limit_upper=1e6,
                target=0.0,
                target_ke=0.0,
                target_kd=0.0,
                armature=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                mode=JointMode.NONE,
            )

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        """
        Initializes a new ModelBuilder instance for constructing simulation models.

        Args:
            up_axis (AxisType, optional): The axis to use as the "up" direction in the simulation.
                Defaults to Axis.Z.
            gravity (float, optional): The magnitude of gravity to apply along the up axis.
                Defaults to -9.81.
        """
        self.num_envs = 0

        # region defaults
        self.default_shape_cfg = ModelBuilder.ShapeConfig()
        self.default_joint_cfg = ModelBuilder.JointDofConfig()

        # Default particle settings
        self.default_particle_radius = 0.1

        # Default triangle soft mesh settings
        self.default_tri_ke = 100.0
        self.default_tri_ka = 100.0
        self.default_tri_kd = 10.0
        self.default_tri_drag = 0.0
        self.default_tri_lift = 0.0

        # Default distance constraint properties
        self.default_spring_ke = 100.0
        self.default_spring_kd = 0.0

        # Default edge bending properties
        self.default_edge_ke = 100.0
        self.default_edge_kd = 0.0

        # Default body settings
        self.default_body_armature = 0.0
        # endregion

        # region compiler settings (similar to MuJoCo)
        self.balance_inertia = True
        """Whether to automatically correct rigid body inertia tensors that violate the triangle inequality.
        When True, adds a scalar multiple of the identity matrix to preserve rotation structure while
        ensuring physical validity (I1 + I2 >= I3 for principal moments). Default: True."""

        self.bound_mass = None
        """Minimum allowed mass value for rigid bodies. If set, any body mass below this value will be
        clamped to this minimum. Set to None to disable mass clamping. Default: None."""

        self.bound_inertia = None
        """Minimum allowed eigenvalue for rigid body inertia tensors. If set, ensures all principal
        moments of inertia are at least this value. Set to None to disable inertia eigenvalue
        clamping. Default: None."""

        self.validate_inertia_detailed = False
        """Whether to use detailed (slower) inertia validation that provides per-body warnings.
        When False, uses a fast GPU kernel that reports only the total number of corrected bodies
        and directly assigns the corrected arrays to the Model (ModelBuilder state is not updated).
        When True, uses a CPU implementation that reports specific issues for each body and updates
        the ModelBuilder's internal state.
        Default: False."""
        # endregion

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_max_velocity = 1e5
        self.particle_color_groups: list[nparray] = []
        self.particle_group = []  # environment group index for each particle

        # shapes (each shape has an entry in these arrays)
        self.shape_key = []  # shape keys
        # transform from shape to body
        self.shape_transform = []
        # maps from shape index to body index
        self.shape_body = []
        self.shape_flags = []
        self.shape_type = []
        self.shape_scale = []
        self.shape_source = []
        self.shape_is_solid = []
        self.shape_thickness = []
        self.shape_material_ke = []
        self.shape_material_kd = []
        self.shape_material_kf = []
        self.shape_material_ka = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        # collision groups within collisions are handled
        self.shape_collision_group = []
        # radius to use for broadphase collision checking
        self.shape_collision_radius = []
        # environment group index for each shape
        self.shape_group = []

        # filtering to ignore certain collision pairs
        self.shape_collision_filter_pairs: list[tuple[int, int]] = []

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []
        self.tri_materials = []
        self.tri_areas = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []
        self.edge_rest_length = []
        self.edge_bending_properties = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activations = []
        self.muscle_bodies = []
        self.muscle_points = []

        # rigid bodies
        self.body_mass = []
        self.body_inertia = []
        self.body_inv_mass = []
        self.body_inv_inertia = []
        self.body_com = []
        self.body_q = []
        self.body_qd = []
        self.body_key = []
        self.body_shapes = {-1: []}  # mapping from body to shapes
        self.body_group = []  # environment group index for each body

        # rigid joints
        self.joint_parent = []  # index of the parent body                      (constant)
        self.joint_parents = {}  # mapping from joint to parent bodies
        self.joint_child = []  # index of the child body                       (constant)
        self.joint_axis = []  # joint axis in child joint frame               (constant)
        self.joint_X_p = []  # frame of joint in parent                      (constant)
        self.joint_X_c = []  # frame of child com (in child coordinates)     (constant)
        self.joint_q = []
        self.joint_qd = []
        self.joint_f = []

        self.joint_type = []
        self.joint_key = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_dof_mode = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_target = []
        self.joint_effort_limit = []
        self.joint_velocity_limit = []
        self.joint_friction = []

        self.joint_twist_lower = []
        self.joint_twist_upper = []

        self.joint_enabled = []

        self.joint_q_start = []
        self.joint_qd_start = []
        self.joint_dof_dim = []
        self.joint_group = []  # environment group index for each joint

        self.articulation_start = []
        self.articulation_key = []
        self.articulation_group = []  # environment group index for each articulation

        self.joint_dof_count = 0
        self.joint_coord_count = 0

        # current environment group index for entities being added directly to this builder.
        # set to -1 to create global entities shared across all environments.
        # note: this value is temporarily overridden when using add_builder().
        self.current_env_group = -1

        self.up_axis: Axis = Axis.from_any(up_axis)
        self.gravity: float = gravity

        # contacts to be generated within the given distance margin to be generated at
        # every simulation substep (can be 0 if only one PBD solver iteration is used)
        self.rigid_contact_margin = 0.1
        # torsional friction coefficient (only considered by XPBD so far)
        self.rigid_contact_torsional_friction = 0.5
        # rolling friction coefficient (only considered by XPBD so far)
        self.rigid_contact_rolling_friction = 0.001

        # number of rigid contact points to allocate in the model during self.finalize() per environment
        # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
        self.num_rigid_contacts_per_env = None

        # equality constraints
        self.equality_constraint_type = []
        self.equality_constraint_body1 = []
        self.equality_constraint_body2 = []
        self.equality_constraint_anchor = []
        self.equality_constraint_relpose = []
        self.equality_constraint_torquescale = []
        self.equality_constraint_joint1 = []
        self.equality_constraint_joint2 = []
        self.equality_constraint_polycoef = []
        self.equality_constraint_key = []
        self.equality_constraint_enabled = []

    @property
    def up_vector(self) -> Vec3:
        """
        Returns the 3D unit vector corresponding to the current up axis (read-only).

        This property computes the up direction as a 3D vector based on the value of :attr:`up_axis`.
        For example, if ``up_axis`` is ``Axis.Z``, this returns ``(0, 0, 1)``.

        Returns:
            Vec3: The 3D up vector corresponding to the current up axis.
        """
        return axis_to_vec3(self.up_axis)

    @up_vector.setter
    def up_vector(self, _):
        raise AttributeError(
            "The 'up_vector' property is read-only and cannot be set. Instead, use 'up_axis' to set the up axis."
        )

    # region counts
    @property
    def shape_count(self):
        """
        The number of shapes in the model.
        """
        return len(self.shape_type)

    @property
    def body_count(self):
        """
        The number of rigid bodies in the model.
        """
        return len(self.body_q)

    @property
    def joint_count(self):
        """
        The number of joints in the model.
        """
        return len(self.joint_type)

    @property
    def particle_count(self):
        """
        The number of particles in the model.
        """
        return len(self.particle_q)

    @property
    def tri_count(self):
        """
        The number of triangles in the model.
        """
        return len(self.tri_poses)

    @property
    def tet_count(self):
        """
        The number of tetrahedra in the model.
        """
        return len(self.tet_poses)

    @property
    def edge_count(self):
        """
        The number of edges (for bending) in the model.
        """
        return len(self.edge_rest_angle)

    @property
    def spring_count(self):
        """
        The number of springs in the model.
        """
        return len(self.spring_rest_length)

    @property
    def muscle_count(self):
        """
        The number of muscles in the model.
        """
        return len(self.muscle_start)

    @property
    def articulation_count(self):
        """
        The number of articulations in the model.
        """
        return len(self.articulation_start)

    # endregion

    def _compute_replicate_offsets(self, num_copies: int, spacing: tuple[float, float, float]):
        # compute positional offsets per environment
        spacing = np.array(spacing, dtype=np.float32)
        nonzeros = np.nonzero(spacing)[0]
        num_dim = nonzeros.shape[0]
        if num_dim > 0:
            side_length = int(np.ceil(num_copies ** (1.0 / num_dim)))
            spacings = []
            if num_dim == 1:
                for i in range(num_copies):
                    spacings.append(i * spacing)
            elif num_dim == 2:
                for i in range(num_copies):
                    d0 = i // side_length
                    d1 = i % side_length
                    offset = np.zeros(3)
                    offset[nonzeros[0]] = d0 * spacing[nonzeros[0]]
                    offset[nonzeros[1]] = d1 * spacing[nonzeros[1]]
                    spacings.append(offset)
            elif num_dim == 3:
                for i in range(num_copies):
                    d0 = i // (side_length * side_length)
                    d1 = (i // side_length) % side_length
                    d2 = i % side_length
                    offset = np.zeros(3)
                    offset[0] = d0 * spacing[0]
                    offset[1] = d1 * spacing[1]
                    offset[2] = d2 * spacing[2]
                    spacings.append(offset)
            spacings = np.array(spacings, dtype=np.float32)
        else:
            spacings = np.zeros((num_copies, 3), dtype=np.float32)
        min_offsets = np.min(spacings, axis=0)
        correction = min_offsets + (np.max(spacings, axis=0) - min_offsets) / 2.0
        # ensure the envs are not shifted below the ground plane
        correction[Axis.from_any(self.up_axis)] = 0.0
        spacings -= correction
        return spacings

    def replicate(
        self,
        builder: ModelBuilder,
        num_copies: int,
        spacing: tuple[float, float, float] = (5.0, 5.0, 0.0),
    ):
        """
        Replicates the given builder multiple times, offsetting each copy according to the supplied spacing.

        This method is useful for creating multiple instances of a sub-model (e.g., robots, environments)
        arranged in a regular grid or along a line. Each copy is offset in space by a multiple of the
        specified spacing vector, and all entities from each copy are assigned to a new environment group.

        Args:
            builder (ModelBuilder): The builder to replicate. All entities from this builder will be copied.
            num_copies (int): The number of copies to create.
            spacing (tuple[float, float, float], optional): The spacing between each copy along each axis.
                For example, (5.0, 5.0, 0.0) arranges copies in a 2D grid in the XY plane.
                Defaults to (5.0, 5.0, 0.0).
        """
        offsets = self._compute_replicate_offsets(num_copies, spacing)
        xform = wp.transform_identity()
        for i in range(num_copies):
            xform[:3] = offsets[i]
            self.add_builder(builder, xform=xform)

    def add_articulation(self, key: str | None = None):
        # an articulation is a set of contiguous bodies bodies from articulation_start[i] to articulation_start[i+1]
        # these are used for computing forward kinematics e.g.:
        # articulations are automatically 'closed' when calling finalize
        self.articulation_start.append(self.joint_count)
        self.articulation_key.append(key or f"articulation_{self.articulation_count}")
        self.articulation_group.append(self.current_env_group)

    # region importers
    def add_urdf(
        self,
        source: str,
        xform: Transform | None = None,
        floating: bool = False,
        base_joint: dict | str | None = None,
        scale: float = 1.0,
        hide_visuals: bool = False,
        parse_visuals_as_colliders: bool = False,
        up_axis: AxisType = Axis.Z,
        force_show_colliders: bool = False,
        enable_self_collisions: bool = True,
        ignore_inertial_definitions: bool = True,
        ensure_nonstatic_links: bool = True,
        static_link_mass: float = 1e-2,
        collapse_fixed_joints: bool = False,
        mesh_maxhullvert: int = MESH_MAXHULLVERT,
    ):
        """
        Parses a URDF file and adds the bodies and joints to the given ModelBuilder.

        Args:
            source (str): The filename of the URDF file to parse.
            xform (Transform): The transform to apply to the root body. If None, the transform is set to identity.
            floating (bool): If True, the root body is a free joint. If False, the root body is connected via a fixed joint to the world, unless a `base_joint` is defined.
            base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
            scale (float): The scaling factor to apply to the imported mechanism.
            hide_visuals (bool): If True, hide visual shapes.
            parse_visuals_as_colliders (bool): If True, the geometry defined under the `<visual>` tags is used for collision handling instead of the `<collision>` geometries.
            up_axis (AxisType): The up axis of the URDF. This is used to transform the URDF to the builder's up axis. It also determines the up axis of capsules and cylinders in the URDF. The default is Z.
            force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
            enable_self_collisions (bool): If True, self-collisions are enabled.
            ignore_inertial_definitions (bool): If True, the inertial parameters defined in the URDF are ignored and the inertia is calculated from the shape geometry.
            ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
            static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
        """
        from ..utils.import_urdf import parse_urdf  # noqa: PLC0415

        return parse_urdf(
            self,
            source,
            xform,
            floating,
            base_joint,
            scale,
            hide_visuals,
            parse_visuals_as_colliders,
            up_axis,
            force_show_colliders,
            enable_self_collisions,
            ignore_inertial_definitions,
            ensure_nonstatic_links,
            static_link_mass,
            collapse_fixed_joints,
            mesh_maxhullvert,
        )

    def add_usd(
        self,
        source,
        xform: Transform | None = None,
        only_load_enabled_rigid_bodies: bool = False,
        only_load_enabled_joints: bool = True,
        joint_drive_gains_scaling: float = 1.0,
        invert_rotations: bool = True,
        verbose: bool = False,
        ignore_paths: list[str] | None = None,
        cloned_env: str | None = None,
        collapse_fixed_joints: bool = False,
        enable_self_collisions: bool = True,
        apply_up_axis_from_stage: bool = False,
        root_path: str = "/",
        joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
        bodies_follow_joint_ordering: bool = True,
        skip_mesh_approximation: bool = False,
        load_non_physics_prims: bool = True,
        hide_collision_shapes: bool = False,
        mesh_maxhullvert: int = MESH_MAXHULLVERT,
    ) -> dict[str, Any]:
        """
        Parses a Universal Scene Description (USD) stage containing UsdPhysics schema definitions for rigid-body articulations and adds the bodies, shapes and joints to the given ModelBuilder.

        The USD description has to be either a path (file name or URL), or an existing USD stage instance that implements the `Stage <https://openusd.org/dev/api/class_usd_stage.html>`_ interface.

        Args:
            source (str | pxr.Usd.Stage): The file path to the USD file, or an existing USD stage instance.
            xform (Transform): The transform to apply to the entire scene.
            only_load_enabled_rigid_bodies (bool): If True, only rigid bodies which do not have `physics:rigidBodyEnabled` set to False are loaded.
            only_load_enabled_joints (bool): If True, only joints which do not have `physics:jointEnabled` set to False are loaded.
            joint_drive_gains_scaling (float): The default scaling of the PD control gains (stiffness and damping), if not set in the PhysicsScene with as "newton:joint_drive_gains_scaling".
            invert_rotations (bool): If True, inverts any rotations defined in the shape transforms.
            verbose (bool): If True, print additional information about the parsed USD file. Default is False.
            ignore_paths (List[str]): A list of regular expressions matching prim paths to ignore.
            cloned_env (str): The prim path of an environment which is cloned within this USD file. Siblings of this environment prim will not be parsed but instead be replicated via `ModelBuilder.add_builder(builder, xform)` to speed up the loading of many instantiated environments.
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged. Only considered if not set on the PhysicsScene as "newton:collapse_fixed_joints".
            enable_self_collisions (bool): Determines the default behavior of whether self-collisions are enabled for all shapes within an articulation. If an articulation has the attribute ``physxArticulation:enabledSelfCollisions`` defined, this attribute takes precedence.
            apply_up_axis_from_stage (bool): If True, the up axis of the stage will be used to set :attr:`newton.ModelBuilder.up_axis`. Otherwise, the stage will be rotated such that its up axis aligns with the builder's up axis. Default is False.
            root_path (str): The USD path to import, defaults to "/".
            joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the USD. Default is "dfs".
            bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the USD. Default is True.
            skip_mesh_approximation (bool): If True, mesh approximation is skipped. Otherwise, meshes are approximated according to the ``physics:approximation`` attribute defined on the UsdPhysicsMeshCollisionAPI (if it is defined). Default is False.
            load_non_physics_prims (bool): If True, prims that are children of a rigid body that do not have a UsdPhysics schema applied are loaded as visual shapes in a separate pass (may slow down the loading process). Otherwise, non-physics prims are ignored. Default is True.
            hide_collision_shapes (bool): If True, collision shapes are hidden. Default is False.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.

        Returns:
            dict: Dictionary with the following entries:

            .. list-table::
                :widths: 25 75

                * - "fps"
                  - USD stage frames per second
                * - "duration"
                  - Difference between end time code and start time code of the USD stage
                * - "up_axis"
                  - :class:`Axis` representing the stage's up axis ("X", "Y", or "Z")
                * - "path_shape_map"
                  - Mapping from prim path (str) of the UsdGeom to the respective shape index in :class:`ModelBuilder`
                * - "path_body_map"
                  - Mapping from prim path (str) of a rigid body prim (e.g. that implements the PhysicsRigidBodyAPI) to the respective body index in :class:`ModelBuilder`
                * - "path_shape_scale"
                  - Mapping from prim path (str) of the UsdGeom to its respective 3D world scale
                * - "mass_unit"
                  - The stage's Kilograms Per Unit (KGPU) definition (1.0 by default)
                * - "linear_unit"
                  - The stage's Meters Per Unit (MPU) definition (1.0 by default)
                * - "scene_attributes"
                  - Dictionary of all attributes applied to the PhysicsScene prim
                * - "collapse_results"
                  - Dictionary returned by :meth:`newton.ModelBuilder.collapse_fixed_joints` if `collapse_fixed_joints` is True, otherwise None.
        """
        from ..utils.import_usd import parse_usd  # noqa: PLC0415

        return parse_usd(
            self,
            source,
            xform,
            only_load_enabled_rigid_bodies,
            only_load_enabled_joints,
            joint_drive_gains_scaling,
            invert_rotations,
            verbose,
            ignore_paths,
            cloned_env,
            collapse_fixed_joints,
            enable_self_collisions,
            apply_up_axis_from_stage,
            root_path,
            joint_ordering,
            bodies_follow_joint_ordering,
            skip_mesh_approximation,
            load_non_physics_prims,
            hide_collision_shapes,
            mesh_maxhullvert,
        )

    def add_mjcf(
        self,
        source: str,
        xform: Transform | None = None,
        floating: bool | None = None,
        base_joint: dict | str | None = None,
        armature_scale: float = 1.0,
        scale: float = 1.0,
        hide_visuals: bool = False,
        parse_visuals_as_colliders: bool = False,
        parse_meshes: bool = True,
        up_axis: AxisType = Axis.Z,
        ignore_names: Sequence[str] = (),
        ignore_classes: Sequence[str] = (),
        visual_classes: Sequence[str] = ("visual",),
        collider_classes: Sequence[str] = ("collision",),
        no_class_as_colliders: bool = True,
        force_show_colliders: bool = False,
        enable_self_collisions: bool = False,
        ignore_inertial_definitions: bool = True,
        ensure_nonstatic_links: bool = True,
        static_link_mass: float = 1e-2,
        collapse_fixed_joints: bool = False,
        verbose: bool = False,
        skip_equality_constraints: bool = False,
        mesh_maxhullvert: int = MESH_MAXHULLVERT,
    ):
        """
        Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.

        Args:
            source (str): The filename of the MuJoCo file to parse, or the MJCF XML string content.
            xform (Transform): The transform to apply to the imported mechanism.
            floating (bool): If True, the articulation is treated as a floating base. If False, the articulation is treated as a fixed base. If None, the articulation is treated as a floating base if a free joint is found in the MJCF, otherwise it is treated as a fixed base.
            base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
            armature_scale (float): Scaling factor to apply to the MJCF-defined joint armature values.
            scale (float): The scaling factor to apply to the imported mechanism.
            hide_visuals (bool): If True, hide visual shapes.
            parse_visuals_as_colliders (bool): If True, the geometry defined under the `visual_classes` tags is used for collision handling instead of the `collider_classes` geometries.
            parse_meshes (bool): Whether geometries of type `"mesh"` should be parsed. If False, geometries of type `"mesh"` are ignored.
            up_axis (AxisType): The up axis of the MuJoCo scene. The default is Z up.
            ignore_names (Sequence[str]): A list of regular expressions. Bodies and joints with a name matching one of the regular expressions will be ignored.
            ignore_classes (Sequence[str]): A list of regular expressions. Bodies and joints with a class matching one of the regular expressions will be ignored.
            visual_classes (Sequence[str]): A list of regular expressions. Visual geometries with a class matching one of the regular expressions will be parsed.
            collider_classes (Sequence[str]): A list of regular expressions. Collision geometries with a class matching one of the regular expressions will be parsed.
            no_class_as_colliders: If True, geometries without a class are parsed as collision geometries. If False, geometries without a class are parsed as visual geometries.
            force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
            enable_self_collisions (bool): If True, self-collisions are enabled.
            ignore_inertial_definitions (bool): If True, the inertial parameters defined in the MJCF are ignored and the inertia is calculated from the shape geometry.
            ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
            static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
            collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
            verbose (bool): If True, print additional information about parsing the MJCF.
            skip_equality_constraints (bool): Whether <equality> tags should be parsed. If True, equality constraints are ignored.
            mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
        """
        from ..utils.import_mjcf import parse_mjcf  # noqa: PLC0415

        return parse_mjcf(
            self,
            source,
            xform,
            floating,
            base_joint,
            armature_scale,
            scale,
            hide_visuals,
            parse_visuals_as_colliders,
            parse_meshes,
            up_axis,
            ignore_names,
            ignore_classes,
            visual_classes,
            collider_classes,
            no_class_as_colliders,
            force_show_colliders,
            enable_self_collisions,
            ignore_inertial_definitions,
            ensure_nonstatic_links,
            static_link_mass,
            collapse_fixed_joints,
            verbose,
            skip_equality_constraints,
            mesh_maxhullvert,
        )

    # endregion

    def add_builder(
        self,
        builder: ModelBuilder,
        xform: Transform | None = None,
        update_num_env_count: bool = True,
        environment: int | None = None,
    ):
        """Copies the data from `builder`, another `ModelBuilder` to this `ModelBuilder`.

        **Environment Group Behavior:**
        When adding a builder, ALL entities from the source builder will be assigned to the same
        environment group, overriding any group assignments that existed in the source builder.
        This ensures that all entities from a sub-builder are grouped together as a single environment.

        Environment groups automatically handle collision filtering between different environments:
        - Entities from different environments (except -1) do not collide with each other
        - Global entities (group -1) collide with all environments
        - Collision groups from the source builder are preserved as-is for fine-grained collision control within each environment

        To create global entities that are shared across all environments, set the main builder's
        `current_env_group` to -1 before adding entities directly (not via add_builder).

        Example::

            main_builder = ModelBuilder()
            # Create global ground plane
            main_builder.current_env_group = -1
            main_builder.add_ground_plane()

            # Create robot builder
            robot_builder = ModelBuilder()
            robot_builder.add_body(...)  # These group assignments will be overridden

            # Add multiple robot instances
            main_builder.add_builder(robot_builder, environment=0)  # All entities -> group 0
            main_builder.add_builder(robot_builder, environment=1)  # All entities -> group 1

        Args:
            builder (ModelBuilder): a model builder to add model data from.
            xform (Transform): offset transform applied to root bodies.
            update_num_env_count (bool): if True, the number of environments is updated appropriately.
                For non-global entities (environment >= 0), this either increments num_envs (when environment is None)
                or ensures num_envs is at least environment+1. Global entities (environment=-1) do not affect num_envs.
            environment (int | None): environment group index to assign to ALL entities from this builder.
                If None, uses the current environment count as the group index. Use -1 for global entities.
                Note: environment=-1 does not increase num_envs even when update_num_env_count=True.
        """

        if builder.up_axis != self.up_axis:
            raise ValueError("Cannot add a builder with a different up axis.")

        # Set the environment group for entities being added
        if environment is None:
            # Use the current environment count as the group index if not specified
            group_idx = self.num_envs if update_num_env_count else self.current_env_group
        else:
            group_idx = environment

        # Save the previous environment group
        prev_env_group = self.current_env_group
        self.current_env_group = group_idx

        # explicitly resolve the transform multiplication function to avoid
        # repeatedly resolving builtin overloads during shape transformation
        transform_mul_cfunc = wp.context.runtime.core.wp_builtin_mul_transformf_transformf

        # dispatches two transform multiplies to the native implementation
        def transform_mul(a, b):
            out = wp.transform.from_buffer(np.empty(7, dtype=np.float32))
            transform_mul_cfunc(a, b, ctypes.byref(out))
            return out

        start_particle_idx = self.particle_count
        if builder.particle_count:
            self.particle_max_velocity = builder.particle_max_velocity
            if xform is not None:
                pos_offset = xform.p
            else:
                pos_offset = np.zeros(3)
            self.particle_q.extend((np.array(builder.particle_q) + pos_offset).tolist())
            # other particle attributes are added below

        if builder.spring_count:
            self.spring_indices.extend((np.array(builder.spring_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.edge_count:
            # Update edge indices by adding offset, preserving -1 values
            edge_indices = np.array(builder.edge_indices, dtype=np.int32)
            mask = edge_indices != -1
            edge_indices[mask] += start_particle_idx
            self.edge_indices.extend(edge_indices.tolist())
        if builder.tri_count:
            self.tri_indices.extend((np.array(builder.tri_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tet_count:
            self.tet_indices.extend((np.array(builder.tet_indices, dtype=np.int32) + start_particle_idx).tolist())

        builder_coloring_translated = [group + start_particle_idx for group in builder.particle_color_groups]
        self.particle_color_groups = combine_independent_particle_coloring(
            self.particle_color_groups, builder_coloring_translated
        )

        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        for s, b in enumerate(builder.shape_body):
            if b > -1:
                new_b = b + start_body_idx
                self.shape_body.append(new_b)
                self.shape_transform.append(builder.shape_transform[s])
            else:
                self.shape_body.append(-1)
                # apply offset transform to root bodies
                if xform is not None:
                    self.shape_transform.append(transform_mul(xform, builder.shape_transform[s]))
                else:
                    self.shape_transform.append(builder.shape_transform[s])

        for b, shapes in builder.body_shapes.items():
            if b == -1:
                self.body_shapes[-1].extend([s + start_shape_idx for s in shapes])
            else:
                self.body_shapes[b + start_body_idx] = [s + start_shape_idx for s in shapes]

        if builder.joint_count:
            start_q = len(self.joint_q)
            start_X_p = len(self.joint_X_p)
            self.joint_X_p.extend(builder.joint_X_p)
            self.joint_q.extend(builder.joint_q)
            if xform is not None:
                for i in range(len(builder.joint_X_p)):
                    if builder.joint_type[i] == JointType.FREE:
                        qi = builder.joint_q_start[i]
                        xform_prev = wp.transform(*builder.joint_q[qi : qi + 7])
                        tf = transform_mul(xform, xform_prev)
                        qi += start_q
                        self.joint_q[qi : qi + 7] = tf
                    elif builder.joint_parent[i] == -1:
                        self.joint_X_p[start_X_p + i] = transform_mul(xform, builder.joint_X_p[i])

            # offset the indices
            self.articulation_start.extend([a + self.joint_count for a in builder.articulation_start])
            self.joint_parent.extend([p + self.body_count if p != -1 else -1 for p in builder.joint_parent])
            self.joint_child.extend([c + self.body_count for c in builder.joint_child])

            self.joint_q_start.extend([c + self.joint_coord_count for c in builder.joint_q_start])
            self.joint_qd_start.extend([c + self.joint_dof_count for c in builder.joint_qd_start])

        if xform is not None:
            for i in range(builder.body_count):
                self.body_q.append(transform_mul(xform, builder.body_q[i]))
        else:
            self.body_q.extend(builder.body_q)

        # Copy collision groups without modification
        self.shape_collision_group.extend(builder.shape_collision_group)

        # Copy collision filter pairs with offset
        self.shape_collision_filter_pairs.extend(
            [(i + start_shape_idx, j + start_shape_idx) for i, j in builder.shape_collision_filter_pairs]
        )

        # Handle environment group assignments
        # For particles
        if builder.particle_count > 0:
            # Override all group indices with current environment group
            particle_groups = [self.current_env_group] * builder.particle_count
            self.particle_group.extend(particle_groups)

        # For bodies
        if builder.body_count > 0:
            body_groups = [self.current_env_group] * builder.body_count
            self.body_group.extend(body_groups)

        # For shapes
        if builder.shape_count > 0:
            shape_groups = [self.current_env_group] * builder.shape_count
            self.shape_group.extend(shape_groups)

        # For joints
        if builder.joint_count > 0:
            joint_groups = [self.current_env_group] * builder.joint_count
            self.joint_group.extend(joint_groups)

        # For articulations
        if builder.articulation_count > 0:
            articulation_groups = [self.current_env_group] * builder.articulation_count
            self.articulation_group.extend(articulation_groups)

        more_builder_attrs = [
            "articulation_key",
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_qd",
            "body_key",
            "joint_type",
            "joint_enabled",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_dof_dim",
            "joint_dof_mode",
            "joint_key",
            "joint_qd",
            "joint_f",
            "joint_target",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_friction",
            "shape_key",
            "shape_flags",
            "shape_type",
            "shape_scale",
            "shape_source",
            "shape_is_solid",
            "shape_thickness",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_collision_radius",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "edge_rest_angle",
            "edge_rest_length",
            "edge_bending_properties",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
            "spring_control",
            "tri_poses",
            "tri_activations",
            "tri_materials",
            "tri_areas",
            "tet_poses",
            "tet_activations",
            "tet_materials",
            "equality_constraint_type",
            "equality_constraint_body1",
            "equality_constraint_body2",
            "equality_constraint_anchor",
            "equality_constraint_torquescale",
            "equality_constraint_relpose",
            "equality_constraint_joint1",
            "equality_constraint_joint2",
            "equality_constraint_polycoef",
            "equality_constraint_key",
            "equality_constraint_enabled",
        ]

        for attr in more_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

        self.joint_dof_count += builder.joint_dof_count
        self.joint_coord_count += builder.joint_coord_count

        if update_num_env_count:
            # Globals do not contribute to the environment count
            if group_idx >= 0:
                # If an explicit environment is provided, ensure num_envs >= group_idx+1.
                # Otherwise, auto-increment for the next environment.
                if environment is None:
                    self.num_envs += 1
                else:
                    self.num_envs = max(self.num_envs, group_idx + 1)

        # Restore the previous environment group
        self.current_env_group = prev_env_group

    def add_body(
        self,
        xform: Transform | None = None,
        armature: float | None = None,
        com: Vec3 | None = None,
        I_m: Mat33 | None = None,
        mass: float = 0.0,
        key: str | None = None,
    ) -> int:
        """Adds a rigid body to the model.

        Args:
            xform: The location of the body in the world frame.
            armature: Artificial inertia added to the body. If None, the default value from :attr:`default_body_armature` is used.
            com: The center of mass of the body w.r.t its origin. If None, the center of mass is assumed to be at the origin.
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass). If None, the inertia tensor is assumed to be zero.
            mass: Mass of the body.
            key: Key of the body (optional).

        Returns:
            The index of the body in the model.

        Note:
            If the mass is zero then the body is treated as kinematic with no dynamics.

        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if com is None:
            com = wp.vec3()
        if I_m is None:
            I_m = wp.mat33()

        body_id = len(self.body_mass)

        # body data
        if armature is None:
            armature = self.default_body_armature
        inertia = I_m + wp.mat33(np.eye(3)) * armature
        self.body_inertia.append(inertia)
        self.body_mass.append(mass)
        self.body_com.append(com)

        if mass > 0.0:
            self.body_inv_mass.append(1.0 / mass)
        else:
            self.body_inv_mass.append(0.0)

        if any(x for x in inertia):
            self.body_inv_inertia.append(wp.inverse(inertia))
        else:
            self.body_inv_inertia.append(inertia)

        self.body_q.append(xform)
        self.body_qd.append(wp.spatial_vector())

        self.body_key.append(key or f"body_{body_id}")
        self.body_shapes[body_id] = []
        self.body_group.append(self.current_env_group)
        return body_id

    # region joints

    def add_joint(
        self,
        joint_type: wp.constant,
        parent: int,
        child: int,
        linear_axes: list[JointDofConfig] | None = None,
        angular_axes: list[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """
        Generic method to add any type of joint to this ModelBuilder.

        Args:
            joint_type (constant): The type of joint to add (see :ref:'joint-types').
            parent (int): The index of the parent body (-1 is the world).
            child (int): The index of the child body.
            linear_axes (list(:class:`JointDofConfig`)): The linear axes (see :class:`JointDofConfig`) of the joint.
            angular_axes (list(:class:`JointDofConfig`)): The angular axes (see :class:`JointDofConfig`) of the joint.
            key (str): The key of the joint (optional).
            parent_xform (Transform): The transform of the joint in the parent body's local frame. If None, the identity transform is used.
            child_xform (Transform): The transform of the joint in the child body's local frame. If None, the identity transform is used.
            collision_filter_parent (bool): Whether to filter collisions between shapes of the parent and child bodies.
            enabled (bool): Whether the joint is enabled (not considered by :class:`SolverFeatherstone`).

        Returns:
            The index of the added joint.
        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        if parent_xform is None:
            parent_xform = wp.transform()
        else:
            parent_xform = wp.transform(*parent_xform)
        if child_xform is None:
            child_xform = wp.transform()
        else:
            child_xform = wp.transform(*child_xform)

        if len(self.articulation_start) == 0:
            # automatically add an articulation if none exists
            self.add_articulation()
        self.joint_type.append(joint_type)
        self.joint_parent.append(parent)
        if child not in self.joint_parents:
            self.joint_parents[child] = [parent]
        else:
            self.joint_parents[child].append(parent)
        self.joint_child.append(child)
        self.joint_X_p.append(wp.transform(parent_xform))
        self.joint_X_c.append(wp.transform(child_xform))
        self.joint_key.append(key or f"joint_{self.joint_count}")
        self.joint_dof_dim.append((len(linear_axes), len(angular_axes)))
        self.joint_enabled.append(enabled)
        self.joint_group.append(self.current_env_group)

        def add_axis_dim(dim: ModelBuilder.JointDofConfig):
            self.joint_axis.append(dim.axis)
            self.joint_dof_mode.append(dim.mode)
            self.joint_target.append(dim.target)
            self.joint_target_ke.append(dim.target_ke)
            self.joint_target_kd.append(dim.target_kd)
            self.joint_limit_ke.append(dim.limit_ke)
            self.joint_limit_kd.append(dim.limit_kd)
            self.joint_armature.append(dim.armature)
            self.joint_effort_limit.append(dim.effort_limit)
            self.joint_velocity_limit.append(dim.velocity_limit)
            self.joint_friction.append(dim.friction)
            if np.isfinite(dim.limit_lower):
                self.joint_limit_lower.append(dim.limit_lower)
            else:
                self.joint_limit_lower.append(-1e6)
            if np.isfinite(dim.limit_upper):
                self.joint_limit_upper.append(dim.limit_upper)
            else:
                self.joint_limit_upper.append(1e6)

        for dim in linear_axes:
            add_axis_dim(dim)
        for dim in angular_axes:
            add_axis_dim(dim)

        dof_count, coord_count = get_joint_dof_count(joint_type, len(linear_axes) + len(angular_axes))

        for _ in range(coord_count):
            self.joint_q.append(0.0)
        for _ in range(dof_count):
            self.joint_qd.append(0.0)
            self.joint_f.append(0.0)

        if joint_type == JointType.FREE or joint_type == JointType.DISTANCE or joint_type == JointType.BALL:
            # ensure that a valid quaternion is used for the angular dofs
            self.joint_q[-1] = 1.0

        self.joint_q_start.append(self.joint_coord_count)
        self.joint_qd_start.append(self.joint_dof_count)

        self.joint_dof_count += dof_count
        self.joint_coord_count += coord_count

        if collision_filter_parent and parent > -1:
            for child_shape in self.body_shapes[child]:
                if not self.shape_flags[child_shape] & ShapeFlags.COLLIDE_SHAPES:
                    continue
                for parent_shape in self.body_shapes[parent]:
                    if not self.shape_flags[parent_shape] & ShapeFlags.COLLIDE_SHAPES:
                        continue
                    # Ensure canonical order (smaller, larger) for consistent lookup
                    a, b = parent_shape, child_shape
                    if a > b:
                        a, b = b, a
                    self.shape_collision_filter_pairs.append((a, b))

        return self.joint_count - 1

    def add_joint_revolute(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig | None = None,
        target: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        mode: int | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a revolute (hinge) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of rotation in the parent body's local frame, can be a :class:`JointDofConfig` object whose settings will be used instead of the other arguments.
            target: The target angle (in radians) or target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            mode: The control mode of the joint. If None, the default value from :attr:`default_joint_control_mode` is used.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force/torque) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target=target if target is not None else self.default_joint_cfg.target,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                mode=mode if mode is not None else self.default_joint_cfg.mode,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
            )
        return self.add_joint(
            JointType.REVOLUTE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_prismatic(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig = Axis.X,
        target: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        mode: int | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a prismatic (sliding) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of rotation in the parent body's local frame, can be a :class:`JointDofConfig` object whose settings will be used instead of the other arguments.
            target: The target angle (in radians) or target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            mode: The control mode of the joint. If None, the default value from :attr:`default_joint_control_mode` is used.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target=target if target is not None else self.default_joint_cfg.target,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                mode=mode if mode is not None else self.default_joint_cfg.mode,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
            )
        return self.add_joint(
            JointType.PRISMATIC,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_ball(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a ball (spherical) joint to the model. Its position is defined by a 4D quaternion (xyzw) and its velocity is a 3D vector.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        return self.add_joint(
            JointType.BALL,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_fixed(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a fixed (static) joint to the model. It has no degrees of freedom.
        See :meth:`collapse_fixed_joints` for a helper function that removes these fixed joints and merges the connecting bodies to simplify the model and improve stability.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint

        """

        return self.add_joint(
            JointType.FIXED,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_free(
        self,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        parent: int = -1,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a free joint to the model.
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (see :ref:`Twist conventions in Newton <Twist conventions>`).
        The positional dofs are initialized by the child body's transform (see :attr:`body_q` and the ``xform`` argument to :meth:`add_body`).

        Args:
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            parent: The index of the parent body (-1 by default to use the world frame, e.g. to make the child body and its children a floating-base mechanism).
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        joint_id = self.add_joint(
            JointType.FREE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            linear_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
        )
        q_start = self.joint_q_start[joint_id]
        # set the positional dofs to the child body's transform
        self.joint_q[q_start : q_start + 7] = list(self.body_q[child])
        return joint_id

    def add_joint_distance(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        min_distance: float = -1.0,
        max_distance: float = 1.0,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a distance joint to the model. The distance joint constraints the distance between the joint anchor points on the two bodies (see :ref:`FK-IK`) it connects to the interval [`min_distance`, `max_distance`].
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (first 3 linear and then 3 angular velocity dimensions).

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            min_distance: The minimum distance between the bodies (no limit if negative).
            max_distance: The maximum distance between the bodies (no limit if negative).
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        .. note:: Distance joints are currently only supported in the :class:`newton.solvers.SolverXPBD`.

        """

        ax = ModelBuilder.JointDofConfig(
            axis=(1.0, 0.0, 0.0),
            limit_lower=min_distance,
            limit_upper=max_distance,
        )
        return self.add_joint(
            JointType.DISTANCE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[
                ax,
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_d6(
        self,
        parent: int,
        child: int,
        linear_axes: Sequence[JointDofConfig] | None = None,
        angular_axes: Sequence[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a generic joint with custom linear and angular axes. The number of axes determines the number of degrees of freedom of the joint.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            linear_axes: A list of linear axes.
            angular_axes: A list of angular axes.
            key: The key of the joint.
            parent_xform (Transform): The transform of the joint in the parent body's local frame
            child_xform (Transform): The transform of the joint in the child body's local frame
            armature: Artificial inertia added around the joint axes. If None, the default value from :attr:`default_joint_armature` is used.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        return self.add_joint(
            JointType.D6,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=list(linear_axes),
            angular_axes=list(angular_axes),
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_equality_constraint(
        self,
        constraint_type: Any,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        torquescale: float | None = None,
        relpose: Transform | None = None,
        joint1: int = -1,
        joint2: int = -1,
        polycoef: list[float] | None = None,
        key: str | None = None,
        enabled: bool = True,
    ) -> int:
        """Generic method to add any type of equality constraint to this ModelBuilder.

        Args:
            constraint_type (constant): Type of constraint ('connect', 'weld', 'joint')
            body1 (int): Index of the first body participating in the constraint (-1 for world)
            body2 (int): Index of the second body participating in the constraint (-1 for world)
            anchor (Vec3): Anchor point on body1
            torquescale (float): Scales the angular residual for weld
            relpose (Transform): Relative pose of body2 for weld. If None, the identity transform is used.
            joint1 (int): Index of the first joint for joint coupling
            joint2 (int): Index of the second joint for joint coupling
            polycoef (list[float]): Polynomial coefficients for joint coupling
            key (str): Optional constraint name
            enabled (bool): Whether constraint is active

        Returns:
            Constraint index
        """

        self.equality_constraint_type.append(constraint_type)
        self.equality_constraint_body1.append(body1)
        self.equality_constraint_body2.append(body2)
        self.equality_constraint_anchor.append(anchor or wp.vec3())
        self.equality_constraint_torquescale.append(torquescale)
        self.equality_constraint_relpose.append(relpose or wp.transform_identity())
        self.equality_constraint_joint1.append(joint1)
        self.equality_constraint_joint2.append(joint2)
        self.equality_constraint_polycoef.append(polycoef or [0.0, 0.0, 0.0, 0.0, 0.0])
        self.equality_constraint_key.append(key)
        self.equality_constraint_enabled.append(enabled)

        return len(self.equality_constraint_type) - 1

    def add_equality_constraint_connect(
        self,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        key: str | None = None,
        enabled: bool = True,
    ) -> int:
        """Adds a connect equality constraint to the model.
        This constraint connects two bodies at a point. It effectively defines a ball joint outside the kinematic tree.

        Args:
            body1: Index of the first body participating in the constraint (-1 for world)
            body2: Index of the second body participating in the constraint (-1 for world)
            anchor: Anchor point on body1
            key: Optional constraint name
            enabled: Whether constraint is active

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.CONNECT,
            body1=body1,
            body2=body2,
            anchor=anchor,
            key=key,
            enabled=enabled,
        )

    def add_equality_constraint_joint(
        self,
        joint1: int = -1,
        joint2: int = -1,
        polycoef: list[float] | None = None,
        key: str | None = None,
        enabled: bool = True,
    ) -> int:
        """Adds a joint equality constraint to the model.
        Constrains the position or angle of one joint to be a quartic polynomial of another joint. Only scalar joint types (prismatic and revolute) can be used.

        Args:
            joint1: Index of the first joint
            joint2: Index of the second joint
            polycoef: Polynomial coefficients for joint coupling
            key: Optional constraint name
            enabled: Whether constraint is active

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.JOINT,
            joint1=joint1,
            joint2=joint2,
            polycoef=polycoef,
            key=key,
            enabled=enabled,
        )

    def add_equality_constraint_weld(
        self,
        body1: int = -1,
        body2: int = -1,
        anchor: Vec3 | None = None,
        torquescale: float | None = None,
        relpose: Transform | None = None,
        key: str | None = None,
        enabled: bool = True,
    ) -> int:
        """Adds a weld equality constraint to the model.
        Attaches two bodies to each other, removing all relative degrees of freedom between them (softly).

        Args:
            body1: Index of the first body participating in the constraint (-1 for world)
            body2: Index of the second body participating in the constraint (-1 for world)
            anchor: Coordinates of the weld point relative to body2
            torquescale: Scales the angular residual for weld
            relpose (Transform): Relative pose of body2 relative to body1. If None, the identity transform is used
            key: Optional constraint name
            enabled: Whether constraint is active

        Returns:
            Constraint index
        """

        return self.add_equality_constraint(
            constraint_type=EqType.WELD,
            body1=body1,
            body2=body2,
            anchor=anchor,
            torquescale=torquescale,
            relpose=relpose,
            key=key,
            enabled=enabled,
        )

    # endregion

    def plot_articulation(
        self,
        show_body_keys=True,
        show_joint_keys=True,
        show_joint_types=True,
        plot_shapes=True,
        show_shape_keys=True,
        show_shape_types=True,
        show_legend=True,
    ):
        """
        Visualizes the model's articulation graph using matplotlib and networkx.
        Uses the spring layout algorithm from networkx to arrange the nodes.
        Bodies are shown as orange squares, shapes are shown as blue circles.

        Args:
            show_body_keys (bool): Whether to show the body keys or indices
            show_joint_keys (bool): Whether to show the joint keys or indices
            show_joint_types (bool): Whether to show the joint types
            plot_shapes (bool): Whether to render the shapes connected to the rigid bodies
            show_shape_keys (bool): Whether to show the shape keys or indices
            show_shape_types (bool): Whether to show the shape geometry types
            show_legend (bool): Whether to show a legend
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import networkx as nx  # noqa: PLC0415

        def joint_type_str(type):
            if type == JointType.FREE:
                return "free"
            elif type == JointType.BALL:
                return "ball"
            elif type == JointType.PRISMATIC:
                return "prismatic"
            elif type == JointType.REVOLUTE:
                return "revolute"
            elif type == JointType.D6:
                return "D6"
            elif type == JointType.FIXED:
                return "fixed"
            elif type == JointType.DISTANCE:
                return "distance"
            return "unknown"

        def shape_type_str(type):
            if type == GeoType.SPHERE:
                return "sphere"
            if type == GeoType.BOX:
                return "box"
            if type == GeoType.CAPSULE:
                return "capsule"
            if type == GeoType.CYLINDER:
                return "cylinder"
            if type == GeoType.CONE:
                return "cone"
            if type == GeoType.MESH:
                return "mesh"
            if type == GeoType.SDF:
                return "sdf"
            if type == GeoType.PLANE:
                return "plane"
            if type == GeoType.NONE:
                return "none"
            return "unknown"

        if show_body_keys:
            vertices = ["world", *self.body_key]
        else:
            vertices = ["-1"] + [str(i) for i in range(self.body_count)]
        if plot_shapes:
            for i in range(self.shape_count):
                shape_label = []
                if show_shape_keys:
                    shape_label.append(self.shape_key[i])
                if show_shape_types:
                    shape_label.append(f"({shape_type_str(self.shape_type[i])})")
                vertices.append("\n".join(shape_label))
        edges = []
        edge_labels = []
        for i in range(self.joint_count):
            edge = (self.joint_child[i] + 1, self.joint_parent[i] + 1)
            edges.append(edge)
            if show_joint_keys:
                joint_label = self.joint_key[i]
            else:
                joint_label = str(i)
            if show_joint_types:
                joint_label += f"\n({joint_type_str(self.joint_type[i])})"
            edge_labels.append(joint_label)

        if plot_shapes:
            for i in range(self.shape_count):
                edges.append((len(self.body_key) + i + 1, self.shape_body[i] + 1))

        # plot graph
        G = nx.Graph()
        for i in range(len(vertices)):
            G.add_node(i, label=vertices[i])
        for i in range(len(edges)):
            label = edge_labels[i] if i < len(edge_labels) else ""
            G.add_edge(edges[i][0], edges[i][1], label=label)
        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos, node_size=0, edgelist=edges[: self.joint_count])
        # render body vertices
        draw_args = {"node_size": 100}
        bodies = nx.subgraph(G, list(range(self.body_count + 1)))
        nx.draw_networkx_nodes(bodies, pos, node_color="orange", node_shape="s", **draw_args)
        if plot_shapes:
            # render shape vertices
            shapes = nx.subgraph(G, list(range(self.body_count + 1, len(vertices))))
            nx.draw_networkx_nodes(shapes, pos, node_color="skyblue", **draw_args)
            nx.draw_networkx_edges(
                G, pos, node_size=0, edgelist=edges[self.joint_count :], edge_color="gray", style="dashed"
            )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6, bbox={"alpha": 0.6, "color": "w", "lw": 0}
        )
        # add node labels
        nx.draw_networkx_labels(G, pos, dict(enumerate(vertices)), font_size=6)
        if show_legend:
            plt.plot([], [], "s", color="orange", label="body")
            plt.plot([], [], "k-", label="joint")
            if plot_shapes:
                plt.plot([], [], "o", color="skyblue", label="shape")
                plt.plot([], [], "k--", label="shape-body connection")
            plt.legend(loc="upper left", fontsize=6)
        plt.show()

    def collapse_fixed_joints(self, verbose=wp.config.verbose):
        """Removes fixed joints from the model and merges the bodies they connect. This is useful for simplifying the model for faster and more stable simulation."""

        body_data = {}
        body_children = {-1: []}
        visited = {}
        merged_body_data = {}
        for i in range(self.body_count):
            key = self.body_key[i]
            body_data[i] = {
                "shapes": self.body_shapes[i],
                "q": self.body_q[i],
                "qd": self.body_qd[i],
                "mass": self.body_mass[i],
                "inertia": wp.mat33(*self.body_inertia[i]),
                "inv_mass": self.body_inv_mass[i],
                "inv_inertia": self.body_inv_inertia[i],
                "com": self.body_com[i],
                "key": key,
                "original_id": i,
            }
            visited[i] = False
            body_children[i] = []

        joint_data = {}
        for i in range(self.joint_count):
            key = self.joint_key[i]
            parent = self.joint_parent[i]
            child = self.joint_child[i]
            body_children[parent].append(child)

            q_start = self.joint_q_start[i]
            qd_start = self.joint_qd_start[i]
            if i < self.joint_count - 1:
                q_dim = self.joint_q_start[i + 1] - q_start
                qd_dim = self.joint_qd_start[i + 1] - qd_start
            else:
                q_dim = len(self.joint_q) - q_start
                qd_dim = len(self.joint_qd) - qd_start

            data = {
                "type": self.joint_type[i],
                "q": self.joint_q[q_start : q_start + q_dim],
                "qd": self.joint_qd[qd_start : qd_start + qd_dim],
                "armature": self.joint_armature[qd_start : qd_start + qd_dim],
                "q_start": q_start,
                "qd_start": qd_start,
                "key": key,
                "parent_xform": wp.transform_expand(self.joint_X_p[i]),
                "child_xform": wp.transform_expand(self.joint_X_c[i]),
                "enabled": self.joint_enabled[i],
                "axes": [],
                "axis_dim": self.joint_dof_dim[i],
                "parent": parent,
                "child": child,
                "original_id": i,
            }
            num_lin_axes, num_ang_axes = self.joint_dof_dim[i]
            for j in range(qd_start, qd_start + num_lin_axes + num_ang_axes):
                data["axes"].append(
                    {
                        "axis": self.joint_axis[j],
                        "axis_mode": self.joint_dof_mode[j],
                        "target_ke": self.joint_target_ke[j],
                        "target_kd": self.joint_target_kd[j],
                        "limit_ke": self.joint_limit_ke[j],
                        "limit_kd": self.joint_limit_kd[j],
                        "limit_lower": self.joint_limit_lower[j],
                        "limit_upper": self.joint_limit_upper[j],
                        "act": self.joint_target[j],
                        "effort_limit": self.joint_effort_limit[j],
                    }
                )

            joint_data[(parent, child)] = data

        # sort body children so we traverse the tree in the same order as the bodies are listed
        for children in body_children.values():
            children.sort(key=lambda x: body_data[x]["original_id"])

        retained_joints = []
        retained_bodies = []
        body_remap = {-1: -1}
        body_merged_parent = {}
        body_merged_transform = {}

        # depth first search over the joint graph
        def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int):
            nonlocal visited
            nonlocal retained_joints
            nonlocal retained_bodies
            nonlocal body_data

            joint = joint_data[(parent_body, child_body)]
            if joint["type"] == JointType.FIXED:
                joint_xform = joint["parent_xform"] * wp.transform_inverse(joint["child_xform"])
                incoming_xform = incoming_xform * joint_xform
                parent_key = self.body_key[parent_body] if parent_body > -1 else "world"
                child_key = self.body_key[child_body]
                last_dynamic_body_key = self.body_key[last_dynamic_body] if last_dynamic_body > -1 else "world"
                if verbose:
                    print(
                        f"Remove fixed joint {joint['key']} between {parent_key} and {child_key}, "
                        f"merging {child_key} into {last_dynamic_body_key}"
                    )
                child_id = body_data[child_body]["original_id"]
                relative_xform = incoming_xform
                merged_body_data[self.body_key[child_body]] = {
                    "relative_xform": relative_xform,
                    "parent_body": self.body_key[parent_body],
                }
                body_merged_parent[child_body] = last_dynamic_body
                body_merged_transform[child_body] = incoming_xform
                for shape in self.body_shapes[child_id]:
                    self.shape_transform[shape] = incoming_xform * self.shape_transform[shape]
                    if verbose:
                        print(
                            f"  Shape {shape} moved to body {last_dynamic_body_key} with transform {self.shape_transform[shape]}"
                        )
                    if last_dynamic_body > -1:
                        self.shape_body[shape] = body_data[last_dynamic_body]["id"]
                        body_data[last_dynamic_body]["shapes"].append(shape)
                    else:
                        self.shape_body[shape] = -1
                        self.body_shapes[-1].append(shape)

                if last_dynamic_body > -1:
                    source_m = body_data[last_dynamic_body]["mass"]
                    source_com = body_data[last_dynamic_body]["com"]
                    # add inertia to last_dynamic_body
                    m = body_data[child_body]["mass"]
                    com = wp.transform_point(incoming_xform, body_data[child_body]["com"])
                    inertia = body_data[child_body]["inertia"]
                    body_data[last_dynamic_body]["inertia"] += transform_inertia(
                        m, inertia, incoming_xform.p, incoming_xform.q
                    )
                    body_data[last_dynamic_body]["mass"] += m
                    body_data[last_dynamic_body]["com"] = (m * com + source_m * source_com) / (m + source_m)
                    # indicate to recompute inverse mass, inertia for this body
                    body_data[last_dynamic_body]["inv_mass"] = None
            else:
                joint["parent_xform"] = incoming_xform * joint["parent_xform"]
                joint["parent"] = last_dynamic_body
                last_dynamic_body = child_body
                incoming_xform = wp.transform()
                retained_joints.append(joint)
                new_id = len(retained_bodies)
                body_data[child_body]["id"] = new_id
                retained_bodies.append(child_body)
                for shape in body_data[child_body]["shapes"]:
                    self.shape_body[shape] = new_id

            visited[parent_body] = True
            if visited[child_body] or child_body not in body_children:
                return
            for child in body_children[child_body]:
                if not visited[child]:
                    dfs(child_body, child, incoming_xform, last_dynamic_body)

        for body in body_children[-1]:
            if not visited[body]:
                dfs(-1, body, wp.transform(), -1)

        # repopulate the model
        # save original body groups before clearing
        original_body_group = self.body_group[:] if self.body_group else []

        self.body_key.clear()
        self.body_q.clear()
        self.body_qd.clear()
        self.body_mass.clear()
        self.body_inertia.clear()
        self.body_com.clear()
        self.body_inv_mass.clear()
        self.body_inv_inertia.clear()
        self.body_group.clear()  # Clear body groups
        static_shapes = self.body_shapes[-1]
        self.body_shapes.clear()
        # restore static shapes
        self.body_shapes[-1] = static_shapes
        for i in retained_bodies:
            body = body_data[i]
            new_id = len(self.body_key)
            body_remap[body["original_id"]] = new_id
            self.body_key.append(body["key"])
            self.body_q.append(body["q"])
            self.body_qd.append(body["qd"])
            m = body["mass"]
            inertia = body["inertia"]
            self.body_mass.append(m)
            self.body_inertia.append(inertia)
            self.body_com.append(body["com"])
            if body["inv_mass"] is None:
                # recompute inverse mass and inertia
                if m > 0.0:
                    self.body_inv_mass.append(1.0 / m)
                    self.body_inv_inertia.append(wp.inverse(inertia))
                else:
                    self.body_inv_mass.append(0.0)
                    self.body_inv_inertia.append(wp.mat33(0.0))
            else:
                self.body_inv_mass.append(body["inv_mass"])
                self.body_inv_inertia.append(body["inv_inertia"])
            self.body_shapes[new_id] = body["shapes"]
            # Rebuild body group - use original group if it exists
            if original_body_group and body["original_id"] < len(original_body_group):
                self.body_group.append(original_body_group[body["original_id"]])
            else:
                # If no group was assigned, use default -1
                self.body_group.append(-1)

        # sort joints so they appear in the same order as before
        retained_joints.sort(key=lambda x: x["original_id"])

        joint_remap = {}
        for i, joint in enumerate(retained_joints):
            joint_remap[joint["original_id"]] = i
        # update articulation_start
        for i, old_i in enumerate(self.articulation_start):
            start_i = old_i
            while start_i not in joint_remap:
                start_i += 1
                if start_i >= self.joint_count:
                    break
            self.articulation_start[i] = joint_remap.get(start_i, start_i)
        # remove empty articulation starts, i.e. where the start and end are the same
        self.articulation_start = list(set(self.articulation_start))

        # save original joint groups before clearing
        original_joint_group = self.joint_group[:] if self.joint_group else []

        self.joint_key.clear()
        self.joint_type.clear()
        self.joint_parent.clear()
        self.joint_child.clear()
        self.joint_q.clear()
        self.joint_qd.clear()
        self.joint_q_start.clear()
        self.joint_qd_start.clear()
        self.joint_enabled.clear()
        self.joint_armature.clear()
        self.joint_X_p.clear()
        self.joint_X_c.clear()
        self.joint_axis.clear()
        self.joint_dof_mode.clear()
        self.joint_target_ke.clear()
        self.joint_target_kd.clear()
        self.joint_limit_lower.clear()
        self.joint_limit_upper.clear()
        self.joint_limit_ke.clear()
        self.joint_effort_limit.clear()
        self.joint_limit_kd.clear()
        self.joint_dof_dim.clear()
        self.joint_target.clear()
        self.joint_group.clear()  # Clear joint groups
        for joint in retained_joints:
            self.joint_key.append(joint["key"])
            self.joint_type.append(joint["type"])
            self.joint_parent.append(body_remap[joint["parent"]])
            self.joint_child.append(body_remap[joint["child"]])
            self.joint_q_start.append(len(self.joint_q))
            self.joint_qd_start.append(len(self.joint_qd))
            self.joint_q.extend(joint["q"])
            self.joint_qd.extend(joint["qd"])
            self.joint_armature.extend(joint["armature"])
            self.joint_enabled.append(joint["enabled"])
            self.joint_X_p.append(joint["parent_xform"])
            self.joint_X_c.append(joint["child_xform"])
            self.joint_dof_dim.append(joint["axis_dim"])
            # Rebuild joint group - use original group if it exists
            if original_joint_group and joint["original_id"] < len(original_joint_group):
                self.joint_group.append(original_joint_group[joint["original_id"]])
            else:
                # If no group was assigned, use default -1
                self.joint_group.append(-1)
            for axis in joint["axes"]:
                self.joint_axis.append(axis["axis"])
                self.joint_dof_mode.append(axis["axis_mode"])
                self.joint_target_ke.append(axis["target_ke"])
                self.joint_target_kd.append(axis["target_kd"])
                self.joint_limit_lower.append(axis["limit_lower"])
                self.joint_limit_upper.append(axis["limit_upper"])
                self.joint_limit_ke.append(axis["limit_ke"])
                self.joint_limit_kd.append(axis["limit_kd"])
                self.joint_target.append(axis["act"])
                self.joint_effort_limit.append(axis["effort_limit"])

        return {
            "body_remap": body_remap,
            "joint_remap": joint_remap,
            "body_merged_parent": body_merged_parent,
            "body_merged_transform": body_merged_transform,
            # TODO clean up this data
            "merged_body_data": merged_body_data,
        }

    # muscles
    def add_muscle(
        self, bodies: list[int], positions: list[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float
    ) -> float:
        """Adds a muscle-tendon activation unit.

        Args:
            bodies: A list of body indices for each waypoint
            positions: A list of positions of each waypoint in the body's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        .. note:: The simulation support for muscles is in progress and not yet fully functional.

        """

        n = len(bodies)

        self.muscle_start.append(len(self.muscle_bodies))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activations.append(0.0)

        for i in range(n):
            self.muscle_bodies.append(bodies[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start) - 1

    # region shapes

    def add_shape(
        self,
        body: int,
        type: int,
        xform: Transform | None = None,
        cfg: ShapeConfig | None = None,
        scale: Vec3 | None = None,
        src: SDF | Mesh | Any | None = None,
        is_static: bool = False,
        key: str | None = None,
    ) -> int:
        """Adds a generic collision shape to the model.

        This is the base method for adding shapes; prefer using specific helpers like :meth:`add_shape_sphere` where possible.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body (e.g., static environment geometry).
            type (int): The geometry type of the shape (e.g., `GeoType.BOX`, `GeoType.SPHERE`).
            xform (Transform | None): The transform of the shape in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            scale (Vec3 | None): The scale of the geometry. The interpretation depends on the shape type. Defaults to `(1.0, 1.0, 1.0)` if `None`.
            src (SDF | Mesh | Any | None): The source geometry data, e.g., a :class:`Mesh` object for `GeoType.MESH` or an :class:`SDF` object for `GeoType.SDF`. Defaults to `None`.
            is_static (bool): If `True`, the shape will have zero mass, and its density property in `cfg` will be effectively ignored for mass calculation. Typically used for fixed, non-movable collision geometry. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated (e.g., "shape_N"). Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if cfg is None:
            cfg = self.default_shape_cfg
        if scale is None:
            scale = (1.0, 1.0, 1.0)
        self.shape_body.append(body)
        shape = self.shape_count
        if cfg.has_shape_collision:
            # no contacts between shapes of the same body
            for same_body_shape in self.body_shapes[body]:
                self.shape_collision_filter_pairs.append((same_body_shape, shape))
        self.body_shapes[body].append(shape)
        self.shape_key.append(key or f"shape_{shape}")
        self.shape_transform.append(xform)
        self.shape_flags.append(cfg.flags)
        self.shape_type.append(type)
        self.shape_scale.append((scale[0], scale[1], scale[2]))
        self.shape_source.append(src)
        self.shape_thickness.append(cfg.thickness)
        self.shape_is_solid.append(cfg.is_solid)
        self.shape_material_ke.append(cfg.ke)
        self.shape_material_kd.append(cfg.kd)
        self.shape_material_kf.append(cfg.kf)
        self.shape_material_ka.append(cfg.ka)
        self.shape_material_mu.append(cfg.mu)
        self.shape_material_restitution.append(cfg.restitution)
        self.shape_collision_group.append(cfg.collision_group)
        self.shape_collision_radius.append(compute_shape_radius(type, scale, src))
        self.shape_group.append(self.current_env_group)
        if cfg.has_shape_collision and cfg.collision_filter_parent and body > -1 and body in self.joint_parents:
            for parent_body in self.joint_parents[body]:
                if parent_body > -1:
                    for parent_shape in self.body_shapes[parent_body]:
                        self.shape_collision_filter_pairs.append((parent_shape, shape))

        if not is_static and cfg.density > 0.0:
            (m, c, I) = compute_shape_inertia(type, scale, src, cfg.density, cfg.is_solid, cfg.thickness)
            com_body = wp.transform_point(xform, c)
            self._update_body_mass(body, m, I, com_body, xform.q)
        return shape

    def add_shape_plane(
        self,
        plane: Vec4 | None = (0.0, 0.0, 1.0, 0.0),
        xform: Transform | None = None,
        width: float = 10.0,
        length: float = 10.0,
        body: int = -1,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """
        Adds a plane collision shape to the model.

        If `xform` is provided, it directly defines the plane's position and orientation. The plane's collision normal
        is assumed to be along the local Z-axis of this `xform`.
        If `xform` is `None`, it will be derived from the `plane` equation `a*x + b*y + c*z + d = 0`.
        Plane shapes added via this method are always static (massless).

        Args:
            plane (Vec4 | None): The plane equation `(a, b, c, d)`. If `xform` is `None`, this defines the plane.
                The normal is `(a,b,c)` and `d` is the offset. Defaults to `(0.0, 0.0, 1.0, 0.0)` (an XY ground plane at Z=0) if `xform` is also `None`.
            xform (Transform | None): The transform of the plane in the world or parent body's frame. If `None`, transform is derived from `plane`. Defaults to `None`.
            width (float): The visual/collision extent of the plane along its local X-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            length (float): The visual/collision extent of the plane along its local Y-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            body (int): The index of the parent body this shape belongs to. Use -1 for world-static planes. Defaults to `-1`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            assert plane is not None, "Either xform or plane must be provided"
            # compute position and rotation from plane equation
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            # compute rotation from local +Z axis to plane normal
            rot = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*normal))
            xform = wp.transform(pos, rot)
        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(width, length, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.PLANE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            is_static=True,
            key=key,
        )

    def add_ground_plane(
        self,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a ground plane collision shape to the model.

        Args:
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        return self.add_shape_plane(
            plane=(*self.up_vector, 0.0),
            width=0.0,
            length=0.0,
            cfg=cfg,
            key=key or "ground_plane",
        )

    def add_shape_sphere(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a sphere collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the sphere in the parent body's local frame. The sphere is centered at this transform's position. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the sphere. Defaults to `1.0`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        scale: Any = wp.vec3(radius, 0.0, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.SPHERE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_box(
        self,
        body: int,
        xform: Transform | None = None,
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a box collision shape to a body.

        The box is centered at its local origin as defined by `xform`.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the box in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            hx (float): The half-extent of the box along its local X-axis. Defaults to `0.5`.
            hy (float): The half-extent of the box along its local Y-axis. Defaults to `0.5`.
            hz (float): The half-extent of the box along its local Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(hx, hy, hz)
        return self.add_shape(
            body=body,
            type=GeoType.BOX,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_capsule(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a capsule collision shape to a body.

        The capsule is centered at its local origin as defined by `xform`. Its length extends along the Z-axis.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the capsule in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the capsule's hemispherical ends and its cylindrical segment. Defaults to `1.0`.
            half_height (float): The half-length of the capsule's central cylindrical segment (excluding the hemispherical ends). Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CAPSULE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_cylinder(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a cylinder collision shape to a body.

        The cylinder is centered at its local origin as defined by `xform`. Its length extends along the Z-axis.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cylinder in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cylinder. Defaults to `1.0`.
            half_height (float): The half-length of the cylinder along the Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CYLINDER,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_cone(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a cone collision shape to a body.

        The cone's origin is at its geometric center, with the base at -half_height and apex at +half_height along the Z-axis.
        The center of mass is located at -half_height/2 from the origin (1/4 of the total height from the base toward the apex).

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cone in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cone's base. Defaults to `1.0`.
            half_height (float): The half-height of the cone (distance from the geometric center to either the base or apex). The total height is 2*half_height. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GeoType.CONE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_mesh(
        self,
        body: int,
        xform: Transform | None = None,
        mesh: Mesh | None = None,
        scale: Vec3 | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a triangle mesh collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the mesh in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            mesh (Mesh | None): The :class:`Mesh` object containing the vertex and triangle data. Defaults to `None`.
            scale (Vec3 | None): The scale of the mesh. Defaults to `None`, in which case the scale is `(1.0, 1.0, 1.0)`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GeoType.MESH,
            xform=xform,
            cfg=cfg,
            scale=scale,
            src=mesh,
            key=key,
        )

    def add_shape_sdf(
        self,
        body: int,
        xform: Transform | None = None,
        sdf: SDF | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a signed distance field (SDF) collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the SDF in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            sdf (SDF | None): The :class:`SDF` object representing the signed distance field. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GeoType.SDF,
            xform=xform,
            cfg=cfg,
            src=sdf,
            key=key,
        )

    def approximate_meshes(
        self,
        method: Literal["coacd", "vhacd", "bounding_sphere", "bounding_box"] | RemeshingMethod = "convex_hull",
        shape_indices: list[int] | None = None,
        raise_on_failure: bool = False,
        **remeshing_kwargs: dict[str, Any],
    ) -> set[int]:
        """Approximates the mesh shapes of the model.

        The following methods are supported:

        +------------------------+-------------------------------------------------------------------------------+
        | Method                 | Description                                                                   |
        +========================+===============================================================================+
        | ``"coacd"``            | Convex decomposition using `CoACD <https://github.com/wjakob/coacd>`_         |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"vhacd"``            | Convex decomposition using `V-HACD <https://github.com/trimesh/vhacdx>`_      |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"bounding_sphere"``  | Approximate the mesh with a sphere                                            |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"bounding_box"``     | Approximate the mesh with an oriented bounding box                            |
        +------------------------+-------------------------------------------------------------------------------+
        | ``"convex_hull"``      | Approximate the mesh with a convex hull (default)                             |
        +------------------------+-------------------------------------------------------------------------------+
        | ``<remeshing_method>`` | Any remeshing method supported by :func:`newton.geometry.remesh_mesh`         |
        +------------------------+-------------------------------------------------------------------------------+

        .. note::

            The ``coacd`` and ``vhacd`` methods require additional dependencies (``coacd`` or ``trimesh`` and ``vhacdx`` respectively) to be installed.
            The convex hull approximation requires ``scipy`` to be installed.

        The ``raise_on_failure`` parameter controls the behavior when the remeshing fails:
            - If `True`, an exception is raised when the remeshing fails.
            - If `False`, a warning is logged, and the method falls back to the next available method in the order of preference:
                - If convex decomposition via CoACD or V-HACD fails or dependencies are not available, the method will fall back to using the ``convex_hull`` method.
                - If convex hull approximation fails, it will fall back to the ``bounding_box`` method.

        Args:
            method: The method to use for approximating the mesh shapes.
            shape_indices: The indices of the shapes to simplify. If `None`, all mesh shapes that have the :attr:`ShapeFlags.COLLIDE_SHAPES` flag set are simplified.
            raise_on_failure: If `True`, raises an exception if the remeshing fails. If `False`, it will log a warning and continue with the fallback method.
            **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

        Returns:
            set[int]: A set of indices of the shapes that were successfully remeshed.
        """
        remeshing_methods = [*RemeshingMethod.__args__, "coacd", "vhacd", "bounding_sphere", "bounding_box"]
        if method not in remeshing_methods:
            raise ValueError(
                f"Unsupported remeshing method: {method}. Supported methods are: {', '.join(remeshing_methods)}."
            )

        if shape_indices is None:
            shape_indices = [
                i
                for i, stype in enumerate(self.shape_type)
                if stype == GeoType.MESH and self.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
            ]

        # keep track of remeshed shapes to handle fallbacks
        remeshed_shapes = set()

        if method == "coacd" or method == "vhacd":
            try:
                if method == "coacd":
                    # convex decomposition using CoACD
                    import coacd  # noqa: PLC0415
                else:
                    # convex decomposition using V-HACD
                    import trimesh  # noqa: PLC0415

                decompositions = {}

                for shape in shape_indices:
                    mesh: Mesh = self.shape_source[shape]
                    scale = self.shape_scale[shape]
                    hash_m = hash(mesh)
                    if hash_m in decompositions:
                        decomposition = decompositions[hash_m]
                    else:
                        if method == "coacd":
                            cmesh = coacd.Mesh(mesh.vertices, mesh.indices.reshape(-1, 3))
                            coacd_settings = {
                                "threshold": 0.5,
                                "mcts_nodes": 20,
                                "mcts_iterations": 5,
                                "mcts_max_depth": 1,
                                "merge": False,
                                "max_convex_hull": mesh.maxhullvert,
                            }
                            coacd_settings.update(remeshing_kwargs)
                            decomposition = coacd.run_coacd(cmesh, **coacd_settings)
                        else:
                            tmesh = trimesh.Trimesh(mesh.vertices, mesh.indices.reshape(-1, 3))
                            vhacd_settings = {
                                "maxNumVerticesPerCH": mesh.maxhullvert,
                            }
                            vhacd_settings.update(remeshing_kwargs)
                            decomposition = trimesh.decomposition.convex_decomposition(tmesh, **vhacd_settings)
                            decomposition = [(d["vertices"], d["faces"]) for d in decomposition]
                        decompositions[hash_m] = decomposition
                    if len(decomposition) == 0:
                        continue
                    # note we need to copy the mesh to avoid modifying the original mesh
                    self.shape_source[shape] = self.shape_source[shape].copy(
                        vertices=decomposition[0][0], indices=decomposition[0][1]
                    )
                    if len(decomposition) > 1:
                        body = self.shape_body[shape]
                        xform = self.shape_transform[shape]
                        cfg = ModelBuilder.ShapeConfig(
                            density=0.0,  # do not add extra mass / inertia
                            ke=self.shape_material_ke[shape],
                            kd=self.shape_material_kd[shape],
                            kf=self.shape_material_kf[shape],
                            ka=self.shape_material_ka[shape],
                            mu=self.shape_material_mu[shape],
                            restitution=self.shape_material_restitution[shape],
                            thickness=self.shape_thickness[shape],
                            is_solid=self.shape_is_solid[shape],
                            collision_group=self.shape_collision_group[shape],
                            collision_filter_parent=self.default_shape_cfg.collision_filter_parent,
                        )
                        cfg.flags = self.shape_flags[shape]
                        for i in range(1, len(decomposition)):
                            self.add_shape_mesh(
                                body=body,
                                xform=xform,
                                cfg=cfg,
                                mesh=Mesh(decomposition[i][0], decomposition[i][1]),
                                key=f"{self.shape_key[shape]}_convex_{i}",
                                scale=scale,
                            )
                    remeshed_shapes.add(shape)
            except Exception as e:
                if raise_on_failure:
                    raise RuntimeError(f"Remeshing with method '{method}' failed.") from e
                else:
                    warnings.warn(
                        f"Remeshing with method '{method}' failed: {e}. Falling back to convex_hull.", stacklevel=2
                    )
                    method = "convex_hull"

        if method in RemeshingMethod.__args__:
            # remeshing of the individual meshes
            remeshed = {}
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    # already remeshed with coacd or vhacd
                    continue
                mesh: Mesh = self.shape_source[shape]
                hash_m = hash(mesh)
                rmesh = remeshed.get(hash_m, None)
                if rmesh is None:
                    try:
                        rmesh = remesh_mesh(mesh, method=method, inplace=False, **remeshing_kwargs)
                        remeshed[hash_m] = rmesh
                    except Exception as e:
                        if raise_on_failure:
                            raise RuntimeError(f"Remeshing with method '{method}' failed for shape {shape}.") from e
                        else:
                            warnings.warn(
                                f"Remeshing with method '{method}' failed for shape {shape}: {e}. Falling back to bounding_box.",
                                stacklevel=2,
                            )
                            continue
                # note we need to copy the mesh to avoid modifying the original mesh
                self.shape_source[shape] = self.shape_source[shape].copy(vertices=rmesh.vertices, indices=rmesh.indices)
                remeshed_shapes.add(shape)

        if method == "bounding_box":
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    continue
                mesh: Mesh = self.shape_source[shape]
                scale = self.shape_scale[shape]
                vertices = mesh.vertices * np.array([*scale])
                tf, scale = compute_obb(vertices)
                self.shape_type[shape] = GeoType.BOX
                self.shape_source[shape] = None
                self.shape_scale[shape] = scale
                shape_tf = wp.transform(*self.shape_transform[shape])
                self.shape_transform[shape] = shape_tf * tf
                remeshed_shapes.add(shape)
        elif method == "bounding_sphere":
            for shape in shape_indices:
                if shape in remeshed_shapes:
                    continue
                mesh: Mesh = self.shape_source[shape]
                scale = self.shape_scale[shape]
                vertices = mesh.vertices * np.array([*scale])
                center = np.mean(vertices, axis=0)
                radius = np.max(np.linalg.norm(vertices - center, axis=1))
                self.shape_type[shape] = GeoType.SPHERE
                self.shape_source[shape] = None
                self.shape_scale[shape] = wp.vec3(radius, 0.0, 0.0)
                tf = wp.transform(center, wp.quat_identity())
                shape_tf = wp.transform(*self.shape_transform[shape])
                self.shape_transform[shape] = shape_tf * tf
                remeshed_shapes.add(shape)

        return remeshed_shapes

    # endregion

    # particles
    def add_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        radius: float | None = None,
        flags: int = ParticleFlags.ACTIVE,
    ) -> int:
        """Adds a single particle to the model.

        Args:
            pos: The initial position of the particle.
            vel: The initial velocity of the particle.
            mass: The mass of the particle.
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.

        Returns:
            The index of the particle in the system.
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        if radius is None:
            radius = self.default_particle_radius
        self.particle_radius.append(radius)
        self.particle_flags.append(flags)
        self.particle_group.append(self.current_env_group)

        particle_id = self.particle_count - 1

        return particle_id

    def add_particles(
        self,
        pos: list[Vec3],
        vel: list[Vec3],
        mass: list[float],
        radius: list[float] | None = None,
        flags: list[wp.uint32] | None = None,
    ):
        """Adds a group particles to the model.

        Args:
            pos: The initial positions of the particle.
            vel: The initial velocities of the particle.
            mass: The mass of the particles.
            radius: The radius of the particles used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particles, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.
        """
        self.particle_q.extend(pos)
        self.particle_qd.extend(vel)
        self.particle_mass.extend(mass)
        if radius is None:
            radius = [self.default_particle_radius] * len(pos)
        if flags is None:
            flags = [ParticleFlags.ACTIVE] * len(pos)
        self.particle_radius.extend(radius)
        self.particle_flags.extend(flags)
        # Maintain environment grouping for bulk particle creation
        self.particle_group.extend([self.current_env_group] * len(pos))

    def add_spring(self, i: int, j, ke: float, kd: float, control: float):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            control: The actuation level of the spring

        Note:
            The spring is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_q[i]
        q = self.particle_q[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

    def add_triangle(
        self,
        i: int,
        j: int,
        k: int,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ) -> float:
        """Adds a triangular FEM element between three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.
        """
        # TODO: Expose elastic parameters on a per-element basis
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        # compute basis for 2D rest pose
        p = self.particle_q[i]
        q = self.particle_q[j]
        r = self.particle_q[k]

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = wp.normalize(wp.cross(qp, rp))
        e1 = wp.normalize(qp)
        e2 = wp.normalize(wp.cross(n, e1))

        R = np.array((e1, e2))
        M = np.array((qp, rp))

        D = R @ M.T

        area = np.linalg.det(D) / 2.0

        if area <= 0.0:
            print("inverted or degenerate triangle element")
            return 0.0
        else:
            inv_D = np.linalg.inv(D)

            self.tri_indices.append((i, j, k))
            self.tri_poses.append(inv_D.tolist())
            self.tri_activations.append(0.0)
            self.tri_materials.append((tri_ke, tri_ka, tri_kd, tri_drag, tri_lift))
            self.tri_areas.append(area)
            return area

    def add_triangles(
        self,
        i: list[int],
        j: list[int],
        k: list[int],
        tri_ke: list[float] | None = None,
        tri_ka: list[float] | None = None,
        tri_kd: list[float] | None = None,
        tri_drag: list[float] | None = None,
        tri_lift: list[float] | None = None,
    ) -> list[float]:
        """Adds triangular FEM elements between groups of three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle

        Return:
            The areas of the triangles

        Note:
            A triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q)[i]
        q = np.array(self.particle_q)[j]
        r = np.array(self.particle_q)[k]

        qp = q - p
        rp = r - p

        def normalized(a):
            l = np.linalg.norm(a, axis=-1, keepdims=True)
            l[l == 0] = 1.0
            return a / l

        n = normalized(np.cross(qp, rp))
        e1 = normalized(qp)
        e2 = normalized(np.cross(n, e1))

        R = np.concatenate((e1[..., None], e2[..., None]), axis=-1)
        M = np.concatenate((qp[..., None], rp[..., None]), axis=-1)

        D = np.matmul(R.transpose(0, 2, 1), M)

        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas > 0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")

        D[areas == 0.0] = np.eye(2)[None, ...]
        inv_D = np.linalg.inv(D)

        inds = np.concatenate((i[valid_inds, None], j[valid_inds, None], k[valid_inds, None]), axis=-1)

        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr

        tri_ke = init_if_none(tri_ke, self.default_tri_ke)
        tri_ka = init_if_none(tri_ka, self.default_tri_ka)
        tri_kd = init_if_none(tri_kd, self.default_tri_kd)
        tri_drag = init_if_none(tri_drag, self.default_tri_drag)
        tri_lift = init_if_none(tri_lift, self.default_tri_lift)

        self.tri_materials.extend(
            zip(
                np.array(tri_ke)[valid_inds],
                np.array(tri_ka)[valid_inds],
                np.array(tri_kd)[valid_inds],
                np.array(tri_drag)[valid_inds],
                np.array(tri_lift)[valid_inds],
                strict=False,
            )
        )
        areas = areas.tolist()
        self.tri_areas.extend(areas)
        return areas

    def add_tetrahedron(
        self, i: int, j: int, k: int, l: int, k_mu: float = 1.0e3, k_lambda: float = 1.0e3, k_damp: float = 0.0
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configuration

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        rest: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
    ) -> None:
        """Adds a bending edge element between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angle across the edge in radians, if not specified it will be computed
            edge_ke: The bending stiffness coefficient
            edge_kd: The bending damping coefficient

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd

        # compute rest angle
        x3 = self.particle_q[k]
        x4 = self.particle_q[l]
        if rest is None:
            rest = 0.0
            if i != -1 and j != -1:
                x1 = self.particle_q[i]
                x2 = self.particle_q[j]

                n1 = wp.normalize(wp.cross(x3 - x1, x4 - x1))
                n2 = wp.normalize(wp.cross(x4 - x2, x3 - x2))
                e = wp.normalize(x4 - x3)

                cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
                sin_theta = np.dot(np.cross(n1, n2), e)
                rest = math.atan2(sin_theta, cos_theta)

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)
        self.edge_rest_length.append(wp.length(x4 - x3))
        self.edge_bending_properties.append((edge_ke, edge_kd))

    def add_edges(
        self,
        i,
        j,
        k,
        l,
        rest: list[float] | None = None,
        edge_ke: list[float] | None = None,
        edge_kd: list[float] | None = None,
    ) -> None:
        """Adds bending edge elements between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angles across the edges in radians, if not specified they will be computed
            edge_ke: The bending stiffness coefficient
            edge_kd: The bending damping coefficient

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        x3 = np.array(self.particle_q)[k]
        x4 = np.array(self.particle_q)[l]
        if rest is None:
            rest = np.zeros_like(i, dtype=float)
            valid_mask = (i != -1) & (j != -1)

            # compute rest angle
            x1_valid = np.array(self.particle_q)[i[valid_mask]]
            x2_valid = np.array(self.particle_q)[j[valid_mask]]
            x3_valid = np.array(self.particle_q)[k[valid_mask]]
            x4_valid = np.array(self.particle_q)[l[valid_mask]]

            def normalized(a):
                l = np.linalg.norm(a, axis=-1, keepdims=True)
                l[l == 0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3_valid - x1_valid, x4_valid - x1_valid))
            n2 = normalized(np.cross(x4_valid - x2_valid, x3_valid - x2_valid))
            e = normalized(x4_valid - x3_valid)

            def dot(a, b):
                return (a * b).sum(axis=-1)

            cos_theta = np.clip(dot(n1, n2), -1.0, 1.0)
            sin_theta = dot(np.cross(n1, n2), e)
            rest[valid_mask] = np.arctan2(sin_theta, cos_theta)

        inds = np.concatenate((i[:, None], j[:, None], k[:, None], l[:, None]), axis=-1)

        self.edge_indices.extend(inds.tolist())
        self.edge_rest_angle.extend(rest.tolist())
        self.edge_rest_length.extend(np.linalg.norm(x4 - x3, axis=1).tolist())

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(i)
            return arr

        edge_ke = init_if_none(edge_ke, self.default_edge_ke)
        edge_kd = init_if_none(edge_kd, self.default_edge_kd)

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd, strict=False))

    def add_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ):
        """Helper to create a regular planar cloth grid

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        indices, vertices = [], []
        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                local_pos = wp.vec3(x * cell_x, y * cell_y, 0.0)
                vertices.append(local_pos)
                if x > 0 and y > 0:
                    v0 = grid_index(x - 1, y - 1, dim_x + 1)
                    v1 = grid_index(x, y - 1, dim_x + 1)
                    v2 = grid_index(x, y, dim_x + 1)
                    v3 = grid_index(x - 1, y, dim_x + 1)
                    if reverse_winding:
                        indices.extend([v0, v1, v2])
                        indices.extend([v0, v2, v3])
                    else:
                        indices.extend([v0, v1, v3])
                        indices.extend([v1, v2, v3])

        start_vertex = len(self.particle_q)

        total_mass = mass * (dim_x + 1) * (dim_x + 1)
        total_area = cell_x * cell_y * dim_x * dim_y
        density = total_mass / total_area

        self.add_cloth_mesh(
            pos=pos,
            rot=rot,
            scale=1.0,
            vel=vel,
            vertices=vertices,
            indices=indices,
            density=density,
            edge_callback=None,
            face_callback=None,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            add_springs=add_springs,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            particle_radius=particle_radius,
        )

        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                particle_mass = mass
                particle_flag = ParticleFlags.ACTIVE

                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    particle_flag = particle_flag & ~ParticleFlags.ACTIVE
                    particle_mass = 0.0

                self.particle_flags[start_vertex + vertex_id] = particle_flag
                self.particle_mass[start_vertex + vertex_id] = particle_mass
                vertex_id = vertex_id + 1

    def add_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        edge_callback=None,
        face_callback=None,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ) -> None:
        """Helper to create a cloth model from a regular triangle mesh

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            edge_callback: A user callback when an edge is created
            face_callback: A user callback when a face is created
            particle_radius: The particle_radius which controls particle based collisions.
        Note:

            The mesh should be two manifold.
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd
        spring_ke = spring_ke if spring_ke is not None else self.default_spring_ke
        spring_kd = spring_kd if spring_kd is not None else self.default_spring_kd
        particle_radius = particle_radius if particle_radius is not None else self.default_particle_radius

        num_verts = int(len(vertices))
        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        # for v in vertices:
        #     p = wp.quat_rotate(rot, v * scale) + pos
        #     self.add_particle(p, vel, 0.0, radius=particle_radius)
        vertices_np = np.array(vertices) * scale
        rot_mat_np = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
        verts_3d_np = np.dot(vertices_np, rot_mat_np.T) + pos
        self.add_particles(
            verts_3d_np.tolist(), [vel] * num_verts, mass=[0.0] * num_verts, radius=[particle_radius] * num_verts
        )

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1, 3)
        areas = self.add_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            [tri_ke] * num_tris,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
        )
        for t in range(num_tris):
            area = areas[t]

            self.particle_mass[inds[t, 0]] += density * area / 3.0
            self.particle_mass[inds[t, 1]] += density * area / 3.0
            self.particle_mass[inds[t, 2]] += density * area / 3.0

        end_tri = len(self.tri_indices)

        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        edge_indices = np.fromiter(
            (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
            int,
        ).reshape(-1, 4)
        self.add_edges(
            edge_indices[:, 0],
            edge_indices[:, 1],
            edge_indices[:, 2],
            edge_indices[:, 3],
            edge_ke=[edge_ke] * len(edge_indices),
            edge_kd=[edge_kd] * len(edge_indices),
        )

        if add_springs:
            spring_indices = set()
            for i, j, k, l in edge_indices:
                spring_indices.add((min(k, l), max(k, l)))
                if i != -1:
                    spring_indices.add((min(i, k), max(i, k)))
                    spring_indices.add((min(i, l), max(i, l)))
                if j != -1:
                    spring_indices.add((min(j, k), max(j, k)))
                    spring_indices.add((min(j, l), max(j, l)))
                if i != -1 and j != -1:
                    spring_indices.add((min(i, j), max(i, j)))

            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    def add_particle_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        mass: float,
        jitter: float,
        radius_mean: float | None = None,
        radius_std: float = 0.0,
    ):
        """
        Adds a regular 3D grid of particles to the model.

        This helper function creates a grid of particles arranged in a rectangular lattice,
        with optional random jitter and per-particle radius variation. The grid is defined
        by its dimensions along each axis and the spacing between particles.

        Args:
            pos (Vec3): The world-space position of the grid origin.
            rot (Quat): The rotation to apply to the grid (as a quaternion).
            vel (Vec3): The initial velocity to assign to each particle.
            dim_x (int): Number of particles along the X axis.
            dim_y (int): Number of particles along the Y axis.
            dim_z (int): Number of particles along the Z axis.
            cell_x (float): Spacing between particles along the X axis.
            cell_y (float): Spacing between particles along the Y axis.
            cell_z (float): Spacing between particles along the Z axis.
            mass (float): Mass to assign to each particle.
            jitter (float): Maximum random offset to apply to each particle position.
            radius_mean (float, optional): Mean radius for particles. If None, uses the builder's default.
            radius_std (float, optional): Standard deviation for particle radii. If > 0, radii are sampled from a normal distribution.

        Returns:
            None
        """
        radius_mean = radius_mean if radius_mean is not None else self.default_particle_radius

        rng = np.random.default_rng(42)
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    p = wp.quat_rotate(rot, v) + pos + wp.vec3(rng.random(3) * jitter)

                    if radius_std > 0.0:
                        r = radius_mean + rng.standard_normal() * radius_std
                    else:
                        r = radius_mean
                    self.add_particle(p, vel, m, r)

    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = wp.quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

    def add_soft_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ) -> None:
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions, array of 3D points
            indices: A list of tetrahedron indices, 4 entries per-element, flattened array
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        pos = wp.vec3(pos[0], pos[1], pos[2])
        # add particles
        for v in vertices:
            p = wp.quat_rotate(rot, wp.vec3(v[0], v[1], v[2]) * scale) + pos

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):
            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if volume > 0.0:
                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add triangles
        for _k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            except np.linalg.LinAlgError:
                continue

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        if i == -1:
            return

        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:  # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = transform_inertia(
            self.body_mass[i], self.body_inertia[i], com_offset, wp.quat_identity()
        ) + transform_inertia(m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

        if new_mass > 0.0:
            self.body_inv_mass[i] = 1.0 / new_mass
        else:
            self.body_inv_mass[i] = 0.0

        if any(x for x in new_inertia):
            self.body_inv_inertia[i] = wp.inverse(new_inertia)
        else:
            self.body_inv_inertia[i] = new_inertia

    def add_free_joints_to_floating_bodies(self, new_bodies: Iterable[int] | None = None):
        """
        Adds a free joint to every rigid body that is not a child in any joint and has positive mass.

        Args:
            new_bodies (Iterable[int] or None, optional): The set of body indices to consider for adding free joints.

        Note:
            - Bodies that are already a child in any joint will be skipped.
            - Only bodies with strictly positive mass will receive a free joint.
            - This is useful for ensuring that all floating (unconnected) bodies are properly articulated.
        """
        # set(self.joint_child) is connected_bodies
        floating_bodies = set(new_bodies) - set(self.joint_child)
        for body_id in floating_bodies:
            if self.body_mass[body_id] > 0:
                self.add_joint_free(child=body_id)

    def set_coloring(self, particle_color_groups):
        """
        Sets coloring information with user-provided coloring.

        Args:
            particle_color_groups: A list of list or `np.array` with `dtype`=`int`. The length of the list is the number of colors
                and each list or `np.array` contains the indices of vertices with this color.
        """
        particle_color_groups = [
            color_group if isinstance(color_group, np.ndarray) else np.array(color_group)
            for color_group in particle_color_groups
        ]
        self.particle_color_groups = particle_color_groups

    def color(
        self,
        include_bending=False,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        coloring_algorithm=ColoringAlgorithm.MCS,
    ):
        """
        Runs coloring algorithm to generate coloring information.

        Args:
            include_bending_energy: Whether to consider bending energy for trimeshes in the coloring process. If set to `True`, the generated
                graph will contain all the edges connecting o1 and o2; otherwise, the graph will be equivalent to the trimesh.
            balance_colors: Whether to apply the color balancing algorithm to balance the size of each color
            target_max_min_color_ratio: the color balancing algorithm will stop when the ratio between the largest color and
                the smallest color reaches this value
            algorithm: Value should be an enum type of ColoringAlgorithm, otherwise it will raise an error. ColoringAlgorithm.mcs means using the MCS coloring algorithm,
                while ColoringAlgorithm.ordered_greedy means using the degree-ordered greedy algorithm. The MCS algorithm typically generates 30% to 50% fewer colors
                compared to the ordered greedy algorithm, while maintaining the same linear complexity. Although MCS has a constant overhead that makes it about twice
                as slow as the greedy algorithm, it produces significantly better coloring results. We recommend using MCS, especially if coloring is only part of the
                preprocessing.

        Note:

            References to the coloring algorithm:

            MCS: Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.

            Ordered Greedy: Ton-That, Q. M., Kry, P. G., & Andrews, S. (2023). Parallel block Neo-Hookean XPBD using graph clustering. Computers & Graphics, 110, 1-10.

        """
        # ignore bending energy if it is too small
        edge_indices = np.array(self.edge_indices)

        self.particle_color_groups = color_trimesh(
            len(self.particle_q),
            edge_indices,
            include_bending,
            algorithm=coloring_algorithm,
            balance_colors=balance_colors,
            target_max_min_color_ratio=target_max_min_color_ratio,
        )

    def finalize(self, device: Devicelike | None = None, requires_grad: bool = False) -> Model:
        """
        Finalize the builder and create a concrete Model for simulation.

        This method transfers all simulation data from the builder to device memory,
        returning a Model object ready for simulation. It should be called after all
        elements (particles, bodies, shapes, joints, etc.) have been added to the builder.

        Args:
            device: The simulation device to use (e.g., 'cpu', 'cuda'). If None, uses the current Warp device.
            requires_grad: If True, enables gradient computation for the model (for differentiable simulation).

        Returns:
            Model: A fully constructed Model object containing all simulation data on the specified device.

        Notes:
            - Performs validation and correction of rigid body inertia and mass properties.
            - Closes all start-index arrays (e.g., for muscles, joints, articulations) with sentinel values.
            - Sets up all arrays and properties required for simulation, including particles, bodies, shapes,
              joints, springs, muscles, constraints, and collision/contact data.
        """
        from .collide import count_rigid_contact_points  # noqa: PLC0415

        # ensure the env count is set correctly
        self.num_envs = max(1, self.num_envs)

        # construct particle inv masses
        ms = np.array(self.particle_mass, dtype=np.float32)
        # static particles (with zero mass) have zero inverse mass
        particle_inv_mass = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)

        with wp.ScopedDevice(device):
            # -------------------------------------
            # construct Model (non-time varying) data

            m = Model(device)
            m.requires_grad = requires_grad

            m.num_envs = self.num_envs

            # ---------------------
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_radius = wp.array(self.particle_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_flags = wp.array([flag_to_int(f) for f in self.particle_flags], dtype=wp.int32)
            m.particle_group = wp.array(self.particle_group, dtype=wp.int32)
            m.particle_max_radius = np.max(self.particle_radius) if len(self.particle_radius) > 0 else 0.0
            m.particle_max_velocity = self.particle_max_velocity

            particle_colors = np.empty(self.particle_count, dtype=int)
            for color in range(len(self.particle_color_groups)):
                particle_colors[self.particle_color_groups[color]] = color
            m.particle_colors = wp.array(particle_colors, dtype=int)
            m.particle_color_groups = [wp.array(group, dtype=int) for group in self.particle_color_groups]

            # hash-grid for particle interactions
            m.particle_grid = wp.HashGrid(128, 128, 128)

            # ---------------------
            # collision geometry

            m.shape_key = self.shape_key
            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_flags = wp.array(self.shape_flags, dtype=wp.int32)
            m.body_shapes = self.body_shapes

            # build list of ids for geometry sources (meshes, sdfs)
            geo_sources = []
            finalized_meshes = {}  # do not duplicate meshes
            for geo in self.shape_source:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if geo:
                    if geo_hash not in finalized_meshes:
                        finalized_meshes[geo_hash] = geo.finalize(device=device)
                    geo_sources.append(finalized_meshes[geo_hash])
                else:
                    # add null pointer
                    geo_sources.append(0)

            m.shape_type = wp.array(self.shape_type, dtype=wp.int32)
            m.shape_source_ptr = wp.array(geo_sources, dtype=wp.uint64)
            m.shape_scale = wp.array(self.shape_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_is_solid = wp.array(self.shape_is_solid, dtype=wp.bool)
            m.shape_thickness = wp.array(self.shape_thickness, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_collision_radius = wp.array(
                self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_group = wp.array(self.shape_group, dtype=wp.int32)

            m.shape_source = self.shape_source  # used for rendering

            m.shape_material_ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_ka = wp.array(self.shape_material_ka, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_material_restitution = wp.array(
                self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad
            )

            m.shape_collision_filter_pairs = set(self.shape_collision_filter_pairs)
            m.shape_collision_group = self.shape_collision_group

            # ---------------------
            # springs

            m.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
            m.spring_rest_length = wp.array(self.spring_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_damping = wp.array(self.spring_damping, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_control = wp.array(self.spring_control, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # triangles

            m.tri_indices = wp.array(self.tri_indices, dtype=wp.int32)
            m.tri_poses = wp.array(self.tri_poses, dtype=wp.mat22, requires_grad=requires_grad)
            m.tri_activations = wp.array(self.tri_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_materials = wp.array(self.tri_materials, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_areas = wp.array(self.tri_areas, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # edges

            m.edge_indices = wp.array(self.edge_indices, dtype=wp.int32)
            m.edge_rest_angle = wp.array(self.edge_rest_angle, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_rest_length = wp.array(self.edge_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = wp.array(
                self.edge_bending_properties, dtype=wp.float32, requires_grad=requires_grad
            )

            # ---------------------
            # tetrahedra

            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, requires_grad=requires_grad)
            m.tet_activations = wp.array(self.tet_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, requires_grad=requires_grad)

            # -----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activations = wp.array(self.muscle_activations, dtype=wp.float32, requires_grad=requires_grad)

            # --------------------------------------
            # rigid bodies

            # Apply inertia verification and correction
            # This catches negative masses/inertias and other critical issues
            if len(self.body_mass) > 0:
                if self.validate_inertia_detailed:
                    # Use detailed Python validation with per-body warnings
                    for i in range(len(self.body_mass)):
                        mass = self.body_mass[i]
                        inertia = self.body_inertia[i]
                        body_key = self.body_key[i] if i < len(self.body_key) else f"body_{i}"

                        corrected_mass, corrected_inertia, was_corrected = verify_and_correct_inertia(
                            mass, inertia, self.balance_inertia, self.bound_mass, self.bound_inertia, body_key
                        )

                        if was_corrected:
                            self.body_mass[i] = corrected_mass
                            self.body_inertia[i] = corrected_inertia
                            # Update inverse mass and inertia
                            if corrected_mass > 0.0:
                                self.body_inv_mass[i] = 1.0 / corrected_mass
                            else:
                                self.body_inv_mass[i] = 0.0

                            if any(x for x in corrected_inertia):
                                self.body_inv_inertia[i] = wp.inverse(corrected_inertia)
                            else:
                                self.body_inv_inertia[i] = corrected_inertia

                    # For detailed validation, create arrays from builder data (which were updated)
                    m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                    m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                    m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                    m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                else:
                    # Use fast Warp kernel validation
                    # First create arrays for the kernel
                    body_mass_array = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                    body_inertia_array = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                    body_inv_mass_array = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                    body_inv_inertia_array = wp.array(
                        self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad
                    )
                    correction_flags = wp.zeros(len(self.body_mass), dtype=wp.bool)

                    # Launch validation kernel
                    wp.launch(
                        kernel=validate_and_correct_inertia_kernel,
                        dim=len(self.body_mass),
                        inputs=[
                            body_mass_array,
                            body_inertia_array,
                            body_inv_mass_array,
                            body_inv_inertia_array,
                            self.balance_inertia,
                            self.bound_mass if self.bound_mass is not None else 0.0,
                            self.bound_inertia if self.bound_inertia is not None else 0.0,
                            correction_flags,
                        ],
                    )

                    # Check if any corrections were made
                    num_corrections = int(np.sum(correction_flags.numpy()))
                    if num_corrections > 0:
                        warnings.warn(
                            f"Inertia validation corrected {num_corrections} bodies. "
                            f"Set validate_inertia_detailed=True for detailed per-body warnings.",
                            stacklevel=2,
                        )

                    # Directly use the corrected arrays on the Model (avoids double allocation)
                    # Note: This means the ModelBuilder's internal state is NOT updated for the fast path
                    m.body_mass = body_mass_array
                    m.body_inv_mass = body_inv_mass_array
                    m.body_inertia = body_inertia_array
                    m.body_inv_inertia = body_inv_inertia_array
            else:
                # No bodies, create empty arrays
                m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
                m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
                m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
                m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)

            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_key = self.body_key
            m.body_group = wp.array(self.body_group, dtype=wp.int32)

            # joints
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_dof_dim = wp.array(np.array(self.joint_dof_dim), dtype=wp.int32, ndim=2)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_key = self.joint_key
            m.joint_group = wp.array(self.joint_group, dtype=wp.int32)
            # compute joint ancestors
            child_to_joint = {}
            for i, child in enumerate(self.joint_child):
                child_to_joint[child] = i
            parent_joint = []
            for parent in self.joint_parent:
                parent_joint.append(child_to_joint.get(parent, -1))
            m.joint_ancestor = wp.array(parent_joint, dtype=wp.int32)

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_dof_mode = wp.array(self.joint_dof_mode, dtype=wp.int32)
            m.joint_target = wp.array(self.joint_target, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_f = wp.array(self.joint_f, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_effort_limit = wp.array(self.joint_effort_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_velocity_limit = wp.array(self.joint_velocity_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_friction = wp.array(self.joint_friction, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_enabled = wp.array(self.joint_enabled, dtype=wp.int32)

            # 'close' the start index arrays with a sentinel value
            joint_q_start = copy.copy(self.joint_q_start)
            joint_q_start.append(self.joint_coord_count)
            joint_qd_start = copy.copy(self.joint_qd_start)
            joint_qd_start.append(self.joint_dof_count)
            articulation_start = copy.copy(self.articulation_start)
            articulation_start.append(self.joint_count)

            # Compute max joints per articulation for IK kernel launches
            max_joints_per_articulation = 0
            for art_idx in range(len(self.articulation_start)):
                joint_start = articulation_start[art_idx]
                joint_end = articulation_start[art_idx + 1]
                num_joints = joint_end - joint_start
                max_joints_per_articulation = max(max_joints_per_articulation, num_joints)

            m.joint_q_start = wp.array(joint_q_start, dtype=wp.int32)
            m.joint_qd_start = wp.array(joint_qd_start, dtype=wp.int32)
            m.articulation_start = wp.array(articulation_start, dtype=wp.int32)
            m.articulation_key = self.articulation_key
            m.articulation_group = wp.array(self.articulation_group, dtype=wp.int32)
            m.max_joints_per_articulation = max_joints_per_articulation

            # equality constraints
            m.equality_constraint_type = wp.array(self.equality_constraint_type, dtype=wp.int32)
            m.equality_constraint_body1 = wp.array(self.equality_constraint_body1, dtype=wp.int32)
            m.equality_constraint_body2 = wp.array(self.equality_constraint_body2, dtype=wp.int32)
            m.equality_constraint_anchor = wp.array(self.equality_constraint_anchor, dtype=wp.vec3)
            m.equality_constraint_torquescale = wp.array(self.equality_constraint_torquescale, dtype=wp.float32)
            m.equality_constraint_relpose = wp.array(
                self.equality_constraint_relpose, dtype=wp.transform, requires_grad=requires_grad
            )
            m.equality_constraint_joint1 = wp.array(self.equality_constraint_joint1, dtype=wp.int32)
            m.equality_constraint_joint2 = wp.array(self.equality_constraint_joint2, dtype=wp.int32)
            m.equality_constraint_polycoef = wp.array(self.equality_constraint_polycoef, dtype=wp.float32)
            m.equality_constraint_key = self.equality_constraint_key
            m.equality_constraint_enabled = wp.array(self.equality_constraint_enabled, dtype=wp.bool)

            # counts
            m.joint_count = self.joint_count
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            m.particle_count = len(self.particle_q)
            m.body_count = len(self.body_q)
            m.shape_count = len(self.shape_type)
            m.tri_count = len(self.tri_poses)
            m.tet_count = len(self.tet_poses)
            m.edge_count = len(self.edge_rest_angle)
            m.spring_count = len(self.spring_rest_length)
            m.muscle_count = len(self.muscle_start)
            m.articulation_count = len(self.articulation_start)
            m.equality_constraint_count = len(self.equality_constraint_type)

            self.find_shape_contact_pairs(m)
            m.rigid_contact_max = count_rigid_contact_points(m)

            m.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            m.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction

            # enable ground plane
            m.up_axis = self.up_axis
            m.up_vector = np.array(self.up_vector, dtype=wp.float32)

            # set gravity
            m.gravity = wp.array(
                [wp.vec3(*(g * self.gravity for g in self.up_vector))],
                dtype=wp.vec3,
                device=device,
                requires_grad=requires_grad,
            )

            return m

    def find_shape_contact_pairs(self, model: Model):
        """
        Identifies and stores all potential shape contact pairs for collision detection.

        This method examines the collision groups and collision masks of all shapes in the model
        to determine which pairs of shapes should be considered for contact generation. It respects
        any user-specified collision filter pairs to avoid redundant or undesired contacts.

        The resulting contact pairs are stored in the model as a 2D array of shape indices.

        Args:
            model (Model): The simulation model to which the contact pairs will be assigned.

        Side Effects:
            - Sets `model.shape_contact_pairs` to a wp.array of shape pairs (wp.vec2i).
            - Sets `model.shape_contact_pair_count` to the number of contact pairs found.
        """
        # Copy the set of filtered-out shape pairs to avoid modifying the original
        filters = model.shape_collision_filter_pairs
        contact_pairs = []

        # Sort shapes by env group in case they are not sorted, keep only colliding shapes
        colliding_indices = [i for i, flag in enumerate(self.shape_flags) if flag & ShapeFlags.COLLIDE_SHAPES]
        sorted_indices = sorted(colliding_indices, key=lambda i: self.shape_group[i])

        # Iterate over all shapes candidates
        for i1 in range(len(sorted_indices)):
            s1 = sorted_indices[i1]
            env1 = self.shape_group[s1]
            collision_group1 = self.shape_collision_group[s1]
            for i2 in range(i1 + 1, len(sorted_indices)):
                s2 = sorted_indices[i2]
                env2 = self.shape_group[s2]
                # Skip shapes from different environments (unless one is global). As the shapes are sorted,
                # this means the shapes in this environment group have all been processed.
                if env1 != -1 and env1 != env2:
                    break

                # Skip shapes from different collision group (unless one is global).
                collision_group2 = self.shape_collision_group[s2]
                if collision_group1 != -1 and collision_group2 != -1 and collision_group1 != collision_group2:
                    continue

                # Ensure canonical order (smaller_element, larger_element)
                if s1 > s2:
                    shape_a, shape_b = s2, s1
                else:
                    shape_a, shape_b = s1, s2

                if (shape_a, shape_b) not in filters:
                    contact_pairs.append((shape_a, shape_b))

        model.shape_contact_pairs = wp.array(np.array(contact_pairs), dtype=wp.vec2i, device=model.device)
        model.shape_contact_pair_count = len(contact_pairs)
