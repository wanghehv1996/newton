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

###########################################################################
# Example IK Franka (positions + rotations)
#
# Inverse kinematics on a Franka FR3 arm targeting the TCP (fr3_hand_tcp).
# - Single IKPositionObjective + IKRotationObjective
# - Gizmo controls the TCP target (with ViewerGL.log_gizmo)
#
# Command: python -m newton.examples ik_franka
###########################################################################

import os
import time

import warp as wp
import numpy as np
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.utils

import io_util
from trajectory_animation import KeyFrameTrajectoryAnimation


def limit_joint_move(tar_q, cur_q, max_qd, dt):
    err = tar_q-cur_q
    max_qd = max_qd*dt
    min_qd = -1 * max_qd
    
    # delta_q = np.clip(err*0.8,min_qd,max_qd)
    delta_q = np.clip(err*0.5,min_qd,max_qd)

    return delta_q, delta_q + cur_q

def transform_diff(tf1, tf2, pos_thres=1e-3, rot_thres=1e-3):
    # Translational distance
    pos1 = wp.transform_get_translation(tf1)
    pos2 = wp.transform_get_translation(tf2)
    trans_dist = wp.length(pos1 - pos2)

    # Rotational distance (angle in radians)
    quat1 = wp.transform_get_rotation(tf1)
    quat2 = wp.transform_get_rotation(tf2)
    dot = wp.dot(quat1, quat2)
    dot = wp.clamp(dot, -1.0, 1.0)  # ensure numerical stability
    rot_dist = 2.0 * wp.acos(abs(dot))  # shortest angle between quaternions

    if trans_dist>pos_thres or rot_dist>rot_thres:
        return True
    return False

# map [0,1] to [low, high]
def linear_map(theta, lo, hi):
    return lo + theta*(hi-lo)


from enum import IntEnum

class GripperControlType(IntEnum):
    """
    Flags for gripper actuator controlling.
    """

    NONE = 0
    """None."""

    TARGET_POSITION = 1
    """Control the gripper finger by setting the target position."""

    TARGET_VELOCITY = 2
    """Control the gripper finger by setting the target velocity."""

class AnimationType(IntEnum):
    """
    Flags for robot animation controlling.
    """

    INTERACTIVE = 0
    """Interactive control with gizmo."""

    TRAJECTORY = 1
    """Trajectory control."""

# config the joint types
fixed_joint_names = {
    "fixed_base", 
    "root_joint", 
    "fl_fixed_joint",
    "fr_fixed_joint",
}
controllable_joint_names = {
    "joint4",
    "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6", 
    "fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6", 
}
gripper_joint_names = {
    "fl_joint7", "fl_joint8",
    "fr_joint7", "fr_joint8",
}

left_ee_body_names = {"fl_link6"}
left_gripper_joint_names = {"fl_joint7", "fl_joint8",}
right_ee_body_names = {"fr_link6"}
right_gripper_joint_names = {"fr_joint7", "fr_joint8",}

# ---------------------------------------------------------------------------
# Warp kernels for GPU-side joint target / velocity updates to eliminate
# repeated host<->device .numpy() transfers inside step().
# ---------------------------------------------------------------------------
@wp.kernel
def precompute_all_substep_targets(
    # Input: current and target transforms for both grippers
    curr_transform_left: wp.transform,
    target_transform_left: wp.transform,
    curr_transform_right: wp.transform,
    target_transform_right: wp.transform,
    # Input: current and target gripper states
    curr_left_state: float,
    target_left_state: float,
    curr_right_state: float,
    target_right_state: float,
    # Input: interpolation parameters
    substeps_total: int,
    # Outputs: pre-computed targets for all substeps (indexed by substep)
    left_pos_targets: wp.array(dtype=wp.vec3),    # type: ignore [substeps]
    left_rot_targets: wp.array(dtype=wp.vec4),    # type: ignore [substeps]
    right_pos_targets: wp.array(dtype=wp.vec3),   # type: ignore [substeps]
    right_rot_targets: wp.array(dtype=wp.vec4),   # type: ignore [substeps]
    left_gripper_states: wp.array(dtype=float),   # type: ignore [substeps]
    right_gripper_states: wp.array(dtype=float),  # type: ignore [substeps]
):
    """
    GPU kernel to pre-compute all substep targets in parallel.
    Runs with dim=substeps_total, one thread per substep.
    """
    substep_idx = wp.tid()
    
    # Compute interpolation ratio for this substep (1-indexed)
    substeps_i = substep_idx + 1
    ratio = float(substeps_i) / float(substeps_total)
    
    # Interpolate left gripper transform
    pos_curr_left = wp.transform_get_translation(curr_transform_left)
    pos_next_left = wp.transform_get_translation(target_transform_left)
    pos_interp_left = pos_curr_left + (pos_next_left - pos_curr_left) * ratio
    
    rot_curr_left = wp.transform_get_rotation(curr_transform_left)
    rot_next_left = wp.transform_get_rotation(target_transform_left)
    rot_interp_left = wp.quat_slerp(rot_curr_left, rot_next_left, ratio)
    
    # Interpolate right gripper transform
    pos_curr_right = wp.transform_get_translation(curr_transform_right)
    pos_next_right = wp.transform_get_translation(target_transform_right)
    pos_interp_right = pos_curr_right + (pos_next_right - pos_curr_right) * ratio
    
    rot_curr_right = wp.transform_get_rotation(curr_transform_right)
    rot_next_right = wp.transform_get_rotation(target_transform_right)
    rot_interp_right = wp.quat_slerp(rot_curr_right, rot_next_right, ratio)
    
    # Store pre-computed targets for this substep
    left_pos_targets[substep_idx] = pos_interp_left
    left_rot_targets[substep_idx] = wp.vec4(rot_interp_left[0], rot_interp_left[1], rot_interp_left[2], rot_interp_left[3])
    
    right_pos_targets[substep_idx] = pos_interp_right
    right_rot_targets[substep_idx] = wp.vec4(rot_interp_right[0], rot_interp_right[1], rot_interp_right[2], rot_interp_right[3])
    
    # Interpolate gripper states
    left_gripper_states[substep_idx] = curr_left_state + (target_left_state - curr_left_state) * ratio
    right_gripper_states[substep_idx] = curr_right_state + (target_right_state - curr_right_state) * ratio


@wp.kernel
def copy_substep_targets_to_ik_objectives(
    substep_idx: int,
    # Pre-computed targets for all substeps
    left_pos_targets: wp.array(dtype=wp.vec3),    # type: ignore [substeps]
    left_rot_targets: wp.array(dtype=wp.vec4),    # type: ignore [substeps]
    right_pos_targets: wp.array(dtype=wp.vec3),   # type: ignore [substeps]
    right_rot_targets: wp.array(dtype=wp.vec4),   # type: ignore [substeps]
    left_gripper_states: wp.array(dtype=float),   # type: ignore [substeps]
    right_gripper_states: wp.array(dtype=float),  # type: ignore [substeps]
    # IK objective targets (single element, will be overwritten each substep)
    ik_left_pos: wp.array(dtype=wp.vec3),         # type: ignore [1]
    ik_left_rot: wp.array(dtype=wp.vec4),         # type: ignore [1]
    ik_right_pos: wp.array(dtype=wp.vec3),        # type: ignore [1]
    ik_right_rot: wp.array(dtype=wp.vec4),        # type: ignore [1]
    out_left_gripper: wp.array(dtype=float),      # type: ignore [1]
    out_right_gripper: wp.array(dtype=float),     # type: ignore [1]
):
    """
    Copy pre-computed targets for a specific substep to IK objective arrays.
    Runs with dim=1.
    """
    ik_left_pos[0] = left_pos_targets[substep_idx]
    ik_left_rot[0] = left_rot_targets[substep_idx]
    ik_right_pos[0] = right_pos_targets[substep_idx]
    ik_right_rot[0] = right_rot_targets[substep_idx]
    out_left_gripper[0] = left_gripper_states[substep_idx]
    out_right_gripper[0] = right_gripper_states[substep_idx]


@wp.kernel
def update_joint_targets(
    state_q: wp.array(dtype=float),            # current joint positions (read)  # type: ignore
    ik_q: wp.array(dtype=float),               # IK joint targets (read)         # type: ignore
    joint_target: wp.array(dtype=float),       # control target array (write)    # type: ignore
    joint_qd: wp.array(dtype=float),           # joint velocity array (write)    # type: ignore
    mask_controllable: wp.array(dtype=int),    # 1 if controllable arm dof       # type: ignore
    mask_left_gripper: wp.array(dtype=int),    # 1 if left gripper dof           # type: ignore
    mask_right_gripper: wp.array(dtype=int),   # 1 if right gripper dof          # type: ignore
    joint_limit_lower: wp.array(dtype=float),  # lower limits                    # type: ignore
    joint_limit_upper: wp.array(dtype=float),  # upper limits                    # type: ignore
    gripper_mode: int,  # 1=TARGET_POSITION, 2=TARGET_VELOCITY
    open_left_array: wp.array(dtype=float),    # left gripper openness [0,1] (GPU array)  # type: ignore
    open_right_array: wp.array(dtype=float),   # right gripper openness [0,1] (GPU array)  # type: ignore
    sim_dt: float,      # substep dt
    substep_i: int,     # current substep index (1-based)
    substeps_total: int, # total substeps
    gripper_vel: float  # max abs velocity for velocity mode
):
    i = wp.tid()

    # Read gripper states directly from GPU arrays (no CPU-GPU sync!)
    open_left = open_left_array[0]
    open_right = open_right_array[0]
    
    # Compute substep ratio in GPU
    substep_ratio = float(substep_i) / float(substeps_total)

    current_q = state_q[i]
    target_full = ik_q[i]

    # Position gripper override: map openness fraction to joint coordinate
    if gripper_mode == 1:  # TARGET_POSITION
        if mask_left_gripper[i] == 1:
            target_full = joint_limit_lower[i] + open_left * (joint_limit_upper[i] - joint_limit_lower[i])
        if mask_right_gripper[i] == 1:
            target_full = joint_limit_lower[i] + open_right * (joint_limit_upper[i] - joint_limit_lower[i])

    # Blend from current toward (possibly overridden) IK target for this substep
    blended = current_q + (target_full - current_q) * substep_ratio
    move = blended - current_q

    # Update control + velocity for active DOFs
    if mask_controllable[i] == 1 or mask_left_gripper[i] == 1 or mask_right_gripper[i] == 1:
        if gripper_mode == 2 and (mask_left_gripper[i] == 1 or mask_right_gripper[i] == 1):  # TARGET_VELOCITY
            # Map openness in [0,1] to velocity in [-gripper_vel, gripper_vel]
            frac = open_left if mask_left_gripper[i] == 1 else open_right
            vel = -gripper_vel + frac * (2.0 * gripper_vel)
            joint_target[i] = vel
            joint_qd[i] = 0.0  # velocity mode doesn't set position target here
        else:
            joint_target[i] = blended
            joint_qd[i] = move / sim_dt



class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # self.gripper_control_type = GripperControlType.TARGET_POSITION # !
        self.gripper_control_type = GripperControlType.TARGET_VELOCITY 

        # TODO: 
        self.use_mujoco_cpu = True  # MuJoCo-CPU is 100x+ faster for cloth simulations!
        # self.use_mujoco_cpu = False # MuJoCo-Warp is VERY slow with cloth (GPU overhead)

        self.animation_type = AnimationType.TRAJECTORY
        # self.animation_type = AnimationType.INTERACTIVE

        # dump visualization image sequence
        self.use_dump_image = False
        # self.use_dump_image = True

        # dump joint q into .npz
        self.use_dump_joint = False

        # Performance debugging
        self.debug_timing = True  # Enable detailed timing output
        self.timing_freq = 10  # Print timing every N frames
        self._detailed_physics_timing = True  # Enable detailed physics breakdown
        
        # Initialize timing member variables
        self._timing_body_q_transfer = 0.0
        self._timing_gripper_transfer = 0.0
        self._timing_trajectory_lookup = 0.0
        self._timing_precompute_kernel = 0.0
        self._timing_clear_forces = 0.0
        self._timing_rigid_solver = 0.0
        self._timing_recover_particles = 0.0
        self._timing_collide = 0.0
        self._timing_cloth_solver = 0.0

        # VBD parameters
        if self.animation_type == AnimationType.INTERACTIVE:
            self.sim_vbd_iterations = 3    
        if self.animation_type == AnimationType.TRAJECTORY:
            self.sim_vbd_iterations = 7
        # self.sim_vbd_iterations = 3
        #       body-cloth contact
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        #       self-contact
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        self.robot_friction = 1.0
        self.table_friction = 0.25
        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6

        self.bending_ke = 1e-4
        self.bending_kd = 1e-3


        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build a single ARX Lift (fixed base) + ground
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        
        franka.add_urdf(
            # lift2 urdf can be downloaded from https://gitee.pjlab.org.cn/L2/wanghui1/PulseAsset.git
            newton.examples.get_asset("lift2_urdf/fixed_robot.urdf"),
            floating=False,
            enable_self_collisions=False,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.17))
        )
        franka.add_ground_plane()

        # ------------------------------------------------------------------
        # Set joint groups, print debug info
        # ------------------------------------------------------------------
        cnt = 0

        # body information
        print("=== Body Information ===")
        for i in range(franka.body_count):
            print(f"body {i}, key={franka.body_key[i]}")
            # set left end effector
            if franka.body_key[i] in left_ee_body_names:
                self.lee_index = i
                print(f"  >> left end-effector")
            # set right end effector
            if franka.body_key[i] in right_ee_body_names:
                self.ree_index = i
                print(f"  >> right end-effector")

        # joint information
        print("=== Joint Information ===")
        print(f"#joint_dof={franka.joint_dof_count}, #joint_coord = {franka.joint_coord_count}")
        # joint groups
        self.fixed_joint_indices = np.array([], dtype = int)
        self.controllable_joint_indices = np.array([], dtype = int)
        self.left_gripper_joint_indices = np.array([], dtype = int)
        self.right_gripper_joint_indices = np.array([], dtype = int)

        for i in range(franka.joint_count):
            print(f"joint {i}, key={franka.joint_key[i]}, type={franka.joint_type[i]}, link={franka.joint_parent[i]} -> {franka.joint_child[i]}, dof_dim={franka.joint_dof_dim[i]}, dof_start {cnt}, dof_lim = [{franka.joint_limit_lower[cnt]}, {franka.joint_limit_upper[cnt]}]")

            dof_start = cnt
            dof_end = cnt + franka.joint_dof_dim[i][0] + franka.joint_dof_dim[i][1]
            # set fixed joint group
            if franka.joint_key[i] in fixed_joint_names:
                for j in range(dof_start, dof_end):
                    self.fixed_joint_indices = np.append(self.fixed_joint_indices, [j])
            # set controllable joint group
            if franka.joint_key[i] in controllable_joint_names:
                for j in range(dof_start, dof_end):
                    self.controllable_joint_indices = np.append(self.controllable_joint_indices, [j])
            # set left gripper joint group
            if franka.joint_key[i] in left_gripper_joint_names:
                for j in range(dof_start, dof_end):
                    self.left_gripper_joint_indices = np.append(self.left_gripper_joint_indices, [j])
            # set right gripper joint group
            if franka.joint_key[i] in right_gripper_joint_names:
                for j in range(dof_start, dof_end):
                    self.right_gripper_joint_indices = np.append(self.right_gripper_joint_indices, [j])

            cnt += franka.joint_dof_dim[i][0] + franka.joint_dof_dim[i][1]

        print(f"joint dq cnt check: {cnt} == {franka.joint_dof_count}")
        print("fixed joint", self.fixed_joint_indices)
        print("controllable joint", self.controllable_joint_indices)
        print("left joint", self.left_gripper_joint_indices)
        print("right joint", self.right_gripper_joint_indices)

        if self.use_dump_joint:
            self.joint_q_seq = np.empty((0, franka.joint_dof_count), dtype=np.float32)
            self.openness_seq = np.empty((0, 2), dtype=np.float32)

        # ------------------------------------------------------------------
        # Configurate joints
        # ------------------------------------------------------------------
        self.robot_joint_q_cnt = len(franka.joint_q)
        
        # Configure target position control for arm joints.
        for i in self.controllable_joint_indices:
            franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            franka.joint_target_ke[i] = 3000.0
            franka.joint_target_kd[i] = 10.0

        # Remove control for the fixed joints
        for i in self.fixed_joint_indices:
            franka.joint_dof_mode[i] = newton.JointMode.NONE
            franka.joint_limit_lower[i] = 0
            franka.joint_limit_upper[i] = 0

        # Configure control for the gripper
        for i in np.concatenate((self.left_gripper_joint_indices, self.right_gripper_joint_indices)):
            # Leave a small gap to avoid penetration
            franka.joint_limit_lower[i] = 0.005
            # franka.joint_limit_upper[i] = 0.04

            # Configure target control for gripper joints
            if self.gripper_control_type == GripperControlType.NONE:
                franka.joint_dof_mode[i] = newton.JointMode.NONE

            if self.gripper_control_type == GripperControlType.TARGET_POSITION:
                franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
                franka.joint_target_ke[i] = 3000.0
                franka.joint_target_kd[i] = 10.0

            if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
                franka.joint_dof_mode[i] = newton.JointMode.TARGET_VELOCITY
                franka.joint_target_kd[i] = 10.0
        
        # ------------------------------------------------------------------
        # Add other objects
        # ------------------------------------------------------------------

        # Add a fixed table
        pos = wp.vec3(1.0, 0.0, 0.201)
        rot = wp.quat_identity()
        body_table = franka.add_body()
        franka.add_joint_fixed(-1, body_table)
        franka.add_shape_box(body_table, xform=wp.transform(p=pos, q=rot), hx=0.6, hy=0.6, hz=0.2)
        # franka.add_shape_cylinder(body_table, xform=wp.transform(p=pos, q=rot), radius=0.4, half_height=0.2)

        # Add a box
        pos = wp.vec3(0.6, 0.0, 0.43)
        rot = wp.quat_identity()
        body_box = franka.add_body(xform=wp.transform(p=pos, q=rot))
        franka.add_joint_free(body_box)
        franka.add_shape_box(body_box, hx=0.03, hy=0.03, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))

        # Set friction
        for i in range(len(franka.shape_material_mu)):
            franka.shape_material_mu[i] = 1.0
            franka.shape_material_ka[i] = 0.002
            franka.shape_is_solid[i] = True

        # Add the T-shirt 
        # garment can be downloaded from https://gitee.pjlab.org.cn/L2/wanghui1/PulseAsset.git
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("PulseAsset/cloth/garment-tri.usdc"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/World/mesh/Mesh"))
        # for prim in usd_stage.Traverse():
        #     print(prim.GetPath())
        #     if prim.IsA(UsdGeom.Mesh):
        #         print("Mesh:", prim.GetPath())
        #     elif prim.IsA(UsdGeom.Points):
        #         print("Points:", prim.GetPath())
        #     elif prim.IsA(UsdGeom.Curves):
        #         print("Curves:", prim.GetPath())
        
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        print("=== Cloth Information ===")
        print(f"vertices = {mesh_points.shape}, faces = {mesh_indices.shape}")

        vertices = [wp.vec3(v) for v in mesh_points]
        franka.add_cloth_mesh(
            vertices=vertices,
            indices=mesh_indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi*0.5),
            pos=wp.vec3(0.7, 0.00, 0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            scale=0.01,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

        franka.color()


        # ------------------------------------------------------------------
        # Finalization and initialization of computational components
        # ------------------------------------------------------------------

        # Finalize builder
        self.model = franka.finalize(requires_grad=False)

        # Set cloth parameter
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        # Warp compute graphs
        self.ik_graph = None
        self.physics_graph = None

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            pos = type(self.viewer.camera.pos)(3.0, 0, 1.4)
            self.viewer.camera.pos = pos
            self.viewer.camera.pitch = -20

        # States
        self.state = self.model.state() # for IKSolver
        self.state_0 = self.model.state() # for Physics Solver
        self.state_1 = self.model.state() # for Physics Solver
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control() # for control

        # ------------------------------------------------------------------
        # End effector
        # ------------------------------------------------------------------
        self.open_left_gripper = 1
        self.open_right_gripper = 1

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state.body_q.numpy()
        self.lee_tf = wp.transform(*body_q_np[self.lee_index])
        self.ree_tf = wp.transform(*body_q_np[self.ree_index])

        # ------------------------------------------------------------------
        # IK setup
        # ------------------------------------------------------------------
        total_residuals = 2 * 6 + self.model.joint_coord_count

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position objective
        self.l_pos_obj = ik.IKPositionObjective(
            link_index=self.lee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.lee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Rotation objective
        self.l_rot_obj = ik.IKRotationObjective(
            link_index=self.lee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.lee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        # Position objective
        self.r_pos_obj = ik.IKPositionObjective(
            link_index=self.ree_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ree_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
        )

        # Rotation objective
        self.r_rot_obj = ik.IKRotationObjective(
            link_index=self.ree_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ree_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=9,
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=12,
            weight=10.0,
        )

        # Variables the solver will update
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.ik_joint_qd = wp.array(self.model.joint_qd, shape=(self.model.joint_dof_count))
        self.ik_iters = 24

        # Flattened view for kernel (avoid per-substep .flatten().numpy() host sync)
        self.ik_joint_q_flat = self.ik_joint_q.flatten()

        # Precompute per-DOF masks as device arrays
        dof_len = self.model.joint_dof_count
        def _make_mask(idxs):
            arr = np.zeros(dof_len, dtype=np.int32)
            for j in idxs:
                if j < dof_len:
                    arr[j] = 1
            return wp.array(arr, dtype=int)

        self.mask_controllable = _make_mask(self.controllable_joint_indices)
        self.mask_left_gripper = _make_mask(self.left_gripper_joint_indices)
        self.mask_right_gripper = _make_mask(self.right_gripper_joint_indices)

        # Gripper state arrays for GPU kernel updates
        self.open_left_gripper_array = wp.array([float(self.open_left_gripper)], dtype=float)
        self.open_right_gripper_array = wp.array([float(self.open_right_gripper)], dtype=float)

        # Pre-computed substep target buffers (for optimized pipeline)
        # Allocate arrays to store all substep targets computed in parallel before the loop
        self.substep_left_pos_targets = wp.zeros(self.sim_substeps, dtype=wp.vec3)
        self.substep_left_rot_targets = wp.zeros(self.sim_substeps, dtype=wp.vec4)
        self.substep_right_pos_targets = wp.zeros(self.sim_substeps, dtype=wp.vec3)
        self.substep_right_rot_targets = wp.zeros(self.sim_substeps, dtype=wp.vec4)
        self.substep_left_gripper_states = wp.zeros(self.sim_substeps, dtype=float)
        self.substep_right_gripper_states = wp.zeros(self.sim_substeps, dtype=float)

        # trajectory animation
        # TODO: better API
        self.trajectory_animation = KeyFrameTrajectoryAnimation()
        self.trajectory_animation.init_lift2_folding()

        # ------------------------------------------------------------------
        # Solvers
        # ------------------------------------------------------------------

        # IK solver
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.ik_joint_q,
            objectives=[self.l_pos_obj, self.l_rot_obj, self.r_pos_obj, self.r_rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.MIXED,
        )

        # Rigid body solver
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=150000, # Increased from 50k - cloth creates many contacts (was getting 105904)
            ncon_per_env=150000, # Increased from 50k - avoid illegal mem access with many cloth contacts
            solver='newton',
            cone="elliptic",
            # disable_contacts=True, 
            use_mujoco_cpu=self.use_mujoco_cpu, # mujoco-cpu or mujoco-warp
            # use_mujoco_contacts=True, # incorrect collision when using mujoco-warp
            use_mujoco_contacts=False, # incorrect friction when using mujoco-warp
            contact_stiffness_time_const=self.sim_dt # important param to ensure zero penetration
        )

        # Cloth solver
        self.model.edge_rest_angle.zero_()
        self.cloth_solver = newton.solvers.SolverVBDPulse(
            self.model,
            iterations=self.sim_vbd_iterations,
            self_contact_radius=self.self_contact_radius,
            self_contact_margin=self.self_contact_margin,
            handle_self_contact=True,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
            integrate_with_external_rigid_solver=True,
            collision_detection_interval=-1,
        )

        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.ik_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            with wp.ScopedCapture() as capture:
                self.ik_simulate()
            self.ik_graph = capture.graph

        self.physics_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            with wp.ScopedCapture() as capture:
                self.physics_simulate()
            self.physics_graph = capture.graph

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            t0 = time.perf_counter()
        self.contacts = self.model.collide(self.state_0)
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            # wp.synchronize()
            t1 = time.perf_counter()
            self._timing_clear_forces = t1 - t0

        # Clear particle info for rigid_solver
        particle_count = self.model.particle_count
        self.model.particle_count = 0
        
        # Step rigid body physics (robot)
        self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            # wp.synchronize()
            t2 = time.perf_counter()
            self._timing_rigid_solver = t2 - t1

        # Recover the particle info
        self.state_0.particle_f.zero_()
        self.model.particle_count = particle_count

        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            # wp.synchronize()
            t3 = time.perf_counter()
            self._timing_recover_particles = t3 - t2

        # Solve the cloth, add force onto state_1
        self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            # wp.synchronize()
            t4 = time.perf_counter()
            self._timing_collide = t4 - t3
        
        self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

        if self.debug_timing and hasattr(self, '_detailed_physics_timing'):
            # wp.synchronize()
            t5 = time.perf_counter()
            self._timing_cloth_solver = t5 - t4

        # swap state
        (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        self.l_pos_obj.set_target_position(0, wp.transform_get_translation(self.lee_tf))
        q = wp.transform_get_rotation(self.lee_tf)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        self.r_pos_obj.set_target_position(0, wp.transform_get_translation(self.ree_tf))
        q = wp.transform_get_rotation(self.ree_tf)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                new_left = 0.0
            else:
                new_left = 1.0

            if self.viewer.is_key_down("2"):
                new_right = 0.0
            else:
                new_right = 1.0
            
            # Update GPU arrays
            self.open_left_gripper_array.assign([new_left])
            self.open_right_gripper_array.assign([new_right])

        print(f"Left  end effector:{self.lee_tf}")
        print(f"Right end effector:{self.ree_tf}")

    def _push_targets_from_trajectories(self):
        """Read transform from trajectory and push into IK objectives."""
        transform, state = self.trajectory_animation.get_pose("left_gripper", self.sim_time)
        transform = wp.transform(*transform)
        self.l_pos_obj.set_target_position(0, wp.transform_get_translation(transform))
        q = wp.transform_get_rotation(transform)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        # Update GPU array
        self.open_left_gripper_array.assign([float(state)])

        transform, state = self.trajectory_animation.get_pose("right_gripper", self.sim_time)
        transform = wp.transform(*transform) # x,y,z + quaternion
        self.r_pos_obj.set_target_position(0, wp.transform_get_translation(transform)) 
        q = wp.transform_get_rotation(transform)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        # Update GPU array
        self.open_right_gripper_array.assign([float(state)])

    def precompute_substep_targets_from_trajectory(self):
        """
        Merged function that:
        1. Reads current gripper transforms and states
        2. Gets target transforms/states from trajectory
        3. Launches GPU kernel to pre-compute all substep targets in parallel
        
        This eliminates redundant host-device transfers and combines operations efficiently.
        """
        if self.debug_timing:
            t0 = time.perf_counter()
        
        # Read current end-effector transforms from physics state
        # Note: FK should be run before first frame to ensure body_q has proper initial transforms
        body_q_np = self.state_0.body_q.numpy()  # CPU-GPU transfer!
        if self.debug_timing:
            t1 = time.perf_counter()
            self._timing_body_q_transfer = t1 - t0
        
        curr_transform_left = wp.transform(*body_q_np[self.lee_index])
        curr_transform_right = wp.transform(*body_q_np[self.ree_index])
        
        # Read current gripper states from GPU arrays
        curr_left_state = float(self.open_left_gripper_array.numpy()[0])  # CPU-GPU transfer!
        curr_right_state = float(self.open_right_gripper_array.numpy()[0])  # CPU-GPU transfer!
        if self.debug_timing:
            t2 = time.perf_counter()
            self._timing_gripper_transfer = t2 - t1
        
        # Get target transforms and states from trajectory animation
        target_transform_left_tuple, target_left_state = self.trajectory_animation.get_pose("left_gripper", self.sim_time)
        target_transform_left = wp.transform(*target_transform_left_tuple)
        
        target_transform_right_tuple, target_right_state = self.trajectory_animation.get_pose("right_gripper", self.sim_time)
        target_transform_right = wp.transform(*target_transform_right_tuple)
        if self.debug_timing:
            t3 = time.perf_counter()
            self._timing_trajectory_lookup = t3 - t2
        
        # Launch GPU kernel to pre-compute all substep targets in parallel
        # This computes interpolated transforms for all substeps at once, maximizing GPU utilization
        wp.launch(
            precompute_all_substep_targets,
            dim=self.sim_substeps,
            inputs=[
                curr_transform_left,
                target_transform_left,
                curr_transform_right,
                target_transform_right,
                float(curr_left_state),
                float(target_left_state),
                float(curr_right_state),
                float(target_right_state),
                self.sim_substeps,
            ],
            outputs=[
                self.substep_left_pos_targets,
                self.substep_left_rot_targets,
                self.substep_right_pos_targets,
                self.substep_right_rot_targets,
                self.substep_left_gripper_states,
                self.substep_right_gripper_states,
            ],
            device=wp.get_device(),
        )
        if self.debug_timing:
            wp.synchronize()  # Force sync to measure GPU time
            t4 = time.perf_counter()
            self._timing_precompute_kernel = t4 - t3

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.debug_timing:
            print("Starting frame timing", self.sim_frame)
            frame_start = time.perf_counter()
            timing_copy = []
            timing_ik = []
            timing_joint_update = []
            timing_physics = []
        
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Pre-compute all substep targets from trajectory in one efficient call
        # This merged function: reads current state, gets trajectory targets, and launches parallel GPU kernel
        self.precompute_substep_targets_from_trajectory()
        
        if self.debug_timing:
            precompute_time = time.perf_counter() - frame_start
        
        # Cache constants outside loop to avoid repeated calculations
        device = wp.get_device()
        gripper_vel = 0.2
        gripper_mode = int(self.gripper_control_type)
        
        # Now execute substep loop using pre-computed targets
        # All operations happen on GPU with minimal CPU-GPU synchronization
        for i in range(self.sim_substeps):
            if self.debug_timing:
                t0 = time.perf_counter()
            
            # Copy pre-computed targets for this substep to IK objectives
            wp.launch(
                copy_substep_targets_to_ik_objectives,
                dim=1,
                inputs=[
                    i,  # substep index (0-based)
                    self.substep_left_pos_targets,
                    self.substep_left_rot_targets,
                    self.substep_right_pos_targets,
                    self.substep_right_rot_targets,
                    self.substep_left_gripper_states,
                    self.substep_right_gripper_states,
                ],
                outputs=[
                    self.l_pos_obj.target_positions,
                    self.l_rot_obj.target_rotations,
                    self.r_pos_obj.target_positions,
                    self.r_rot_obj.target_rotations,
                    self.open_left_gripper_array,
                    self.open_right_gripper_array,
                ],
                device=device,
            )
            
            if self.debug_timing:
                wp.synchronize()
                t1 = time.perf_counter()
                timing_copy.append(t1 - t0)

            # Solve IK (GPU if captured graph)
            if self.ik_graph:
                # print("Launching IK graph")
                wp.capture_launch(self.ik_graph)
            else:
                # print('Launching IK simulate')
                self.ik_simulate()
            
            if self.debug_timing:
                wp.synchronize()
                t2 = time.perf_counter()
                timing_ik.append(t2 - t1)

            # Update joint targets on GPU
            # Substep ratio and gripper states are computed/read directly in GPU kernel
            wp.launch(
                update_joint_targets,
                dim=self.model.joint_dof_count,
                inputs=[
                    self.state_0.joint_q,
                    self.ik_joint_q_flat,
                    self.control.joint_target,
                    self.state_0.joint_qd,
                    self.mask_controllable,
                    self.mask_left_gripper,
                    self.mask_right_gripper,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    gripper_mode,
                    self.open_left_gripper_array,
                    self.open_right_gripper_array,
                    float(self.sim_dt),
                    i + 1,  # substep_i (1-based)
                    self.sim_substeps,
                    gripper_vel,
                ],
                device=device,
            )
            
            if self.debug_timing:
                wp.synchronize()
                t3 = time.perf_counter()
                timing_joint_update.append(t3 - t2)
            
            # Run physics simulation

            if self.physics_graph:
                # print("Launching Physics graph")
                wp.capture_launch(self.physics_graph)
            else:
                self.physics_simulate()
            
            if self.debug_timing:
                wp.synchronize()
                t4 = time.perf_counter()
                timing_physics.append(t4 - t3)

        self.sim_time += self.frame_dt
        self.sim_frame += 1
        
        # Print detailed timing information
        if self.debug_timing and (self.sim_frame % self.timing_freq == 0):
            frame_total = time.perf_counter() - frame_start
            print(f"\n{'='*80}")
            print(f"Frame {self.sim_frame} Performance Breakdown:")
            print(f"{'='*80}")
            print(f"Precompute Phase:")
            print(f"  - Body Q Transfer (CPU←GPU):    {self._timing_body_q_transfer*1000:6.2f} ms  ⚠️ CPU-GPU sync")
            print(f"  - Gripper Transfer (CPU←GPU):   {self._timing_gripper_transfer*1000:6.2f} ms  ⚠️ CPU-GPU sync")
            print(f"  - Trajectory Lookup:            {self._timing_trajectory_lookup*1000:6.2f} ms")
            print(f"  - GPU Precompute Kernel:        {self._timing_precompute_kernel*1000:6.2f} ms")
            print(f"  - Precompute Total:             {precompute_time*1000:6.2f} ms")
            print(f"\nSubstep Loop ({self.sim_substeps} substeps):")
            print(f"  - Copy Targets (avg):           {np.mean(timing_copy)*1000:6.2f} ms")
            print(f"  - IK Solve (avg):               {np.mean(timing_ik)*1000:6.2f} ms")
            print(f"  - Joint Update (avg):           {np.mean(timing_joint_update)*1000:6.2f} ms")
            print(f"  - Physics (avg):                {np.mean(timing_physics)*1000:6.2f} ms")
            
            # Detailed physics breakdown if enabled
            if self._detailed_physics_timing:
                print(f"\n  Physics Breakdown (last substep):")
                print(f"    • Clear Forces:               {self._timing_clear_forces*1000:7.2f} ms")
                print(f"    • Rigid Solver (MuJoCo):      {self._timing_rigid_solver*1000:7.2f} ms  ⚠️ BOTTLENECK!")
                print(f"    • Recover Particles:          {self._timing_recover_particles*1000:7.2f} ms")
                print(f"    • Collide (cloth-body):       {self._timing_collide*1000:7.2f} ms")
                print(f"    • Cloth Solver (VBD):         {self._timing_cloth_solver*1000:7.2f} ms")
                total_physics = (self._timing_clear_forces + self._timing_rigid_solver + 
                               self._timing_recover_particles + self._timing_collide + self._timing_cloth_solver)
                print(f"    • Total:                      {total_physics*1000:7.2f} ms")
            
            print(f"\n  - Per-Substep Total (avg):      {(np.mean(timing_copy) + np.mean(timing_ik) + np.mean(timing_joint_update) + np.mean(timing_physics))*1000:6.2f} ms")
            print(f"  - All Substeps Total:           {(sum(timing_copy) + sum(timing_ik) + sum(timing_joint_update) + sum(timing_physics))*1000:6.2f} ms")
            print(f"\nFrame Total:                      {frame_total*1000:6.2f} ms ({1.0/frame_total:.1f} FPS)")
            print(f"Target Frame Time (60 FPS):       {1000.0/60.0:.2f} ms")
            
            # Add recommendation based on timing
            if self._timing_rigid_solver > 0.1:  # If rigid solver takes more than 100ms
                print(f"\n⚠️  WARNING: MuJoCo-Warp rigid solver is very slow ({self._timing_rigid_solver*1000:.1f} ms)")
                print(f"   SOLUTION: Set use_mujoco_cpu=True for better performance with cloth!")
            if frame_total > 1.0/60.0:
                print(f"⚠️  Running {frame_total/(1.0/60.0):.2f}x slower than real-time!")
            print(f"{'='*80}\n")
        
        if self.use_dump_joint:
            joint_q_np = self.state_0.joint_q.numpy()
            self.joint_q_seq = np.vstack((self.joint_q_seq, joint_q_np[0:self.robot_joint_q_cnt]))
            # Read final gripper states from GPU arrays
            open_left = self.open_left_gripper_array.numpy()[0]
            open_right = self.open_right_gripper_array.numpy()[0]
            self.openness_seq = np.vstack((self.openness_seq, np.array([open_left, open_right])))

            if self.sim_frame == 32 * self.fps:
                np.savez('lift2_manipulating_cloth.npz', joint_q=self.joint_q_seq, openness=self.openness_seq)


    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            # Register gizmo (viewer will draw & mutate transform in-place)
            self.viewer.log_gizmo("left_target_tcp", self.lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.ree_tf)
        # self.viewer.log_state(self.state)
        self.viewer.log_state(self.state_0)

        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        wp.synchronize()

        if self.use_dump_image:
            io_util.dump_gl_frame_image(self.viewer.renderer._screen_width,self.viewer.renderer._screen_height,f"img_{self.sim_frame}.png")

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    if not os.environ.get("DISPLAY"):
        parser.set_defaults(viewer="null", headless=True)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
