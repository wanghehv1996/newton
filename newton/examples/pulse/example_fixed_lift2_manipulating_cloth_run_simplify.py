import warp as wp
import numpy as np
from pxr import Usd, UsdGeom
import time

import newton
import newton.examples
import newton.ik as ik
import newton.utils

import io_util
from trajectory_animation import KeyFrameTrajectoryAnimation
import os
from enum import IntEnum

@wp.kernel
def _update_control_kernel(
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    left_indices: wp.array(dtype=int),
    right_indices: wp.array(dtype=int),
    controllable_indices: wp.array(dtype=int),
    q0_frame: wp.array(dtype=float),
    q_target_frame: wp.array(dtype=float),
    current_q: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    ik_joint_qd: wp.array(dtype=float),
    state_qd: wp.array(dtype=float),
    left_count: int,
    right_count: int,
    controllable_count: int,
    sim_substeps: int,
    substep_index: int,
    sim_dt: float,
    gripper_params: wp.array(dtype=float),  # [left_prev, left_target, right_prev, right_target]
    gripper_control_type: int,
):
    tid = wp.tid()

    t = float(substep_index + 1) / float(sim_substeps)
    left_prev = gripper_params[0]
    left_target = gripper_params[1]
    right_prev = gripper_params[2]
    right_target = gripper_params[3]
    # Asymmetric openness rule:
    # - 0 -> 1: switch immediately (no interpolation)
    # - 1 -> 0: interpolate linearly over substeps
    if left_prev < left_target:
        left_open = left_target
    elif left_prev > left_target:
        left_open = (1.0 - t) * left_prev + t * left_target
    else:
        left_open = left_prev

    if right_prev < right_target:
        right_open = right_target
    elif right_prev > right_target:
        right_open = (1.0 - t) * right_prev + t * right_target
    else:
        right_open = right_prev

    if gripper_control_type == 1:
        if tid < left_count:
            li = left_indices[tid]
            lo = joint_limit_lower[li]
            hi = joint_limit_upper[li]
            q_target_frame[li] = lo + left_open * (hi - lo)

            alpha = float(substep_index + 1) / float(sim_substeps)
            target = q0_frame[li] + alpha * (q_target_frame[li] - q0_frame[li])
            v = (target - current_q[li]) / sim_dt
            joint_target[li] = target
            ik_joint_qd[li] = v
            state_qd[li] = v

        if tid < right_count:
            ri = right_indices[tid]
            lo_r = joint_limit_lower[ri]
            hi_r = joint_limit_upper[ri]
            q_target_frame[ri] = lo_r + right_open * (hi_r - lo_r)

            alpha = float(substep_index + 1) / float(sim_substeps)
            target_r = q0_frame[ri] + alpha * (q_target_frame[ri] - q0_frame[ri])
            v_r = (target_r - current_q[ri]) / sim_dt
            joint_target[ri] = target_r
            ik_joint_qd[ri] = v_r
            state_qd[ri] = v_r

    elif gripper_control_type == 2:
        # Velocity control: compute per-substep velocity targets towards the
        # frame's openness goal, splitting across sim_substeps
        if tid < left_count:
            li = left_indices[tid]
            lo = joint_limit_lower[li]
            hi = joint_limit_upper[li]
            # End-of-frame target position from openness
            q_end = lo + left_open * (hi - lo)
            # Store for reference (not used by velocity mode actuation directly)
            q_target_frame[li] = q_end

            alpha = float(substep_index + 1) / float(sim_substeps)
            target = q0_frame[li] + alpha * (q_end - q0_frame[li])
            v = (target - current_q[li]) / sim_dt
            joint_target[li] = v
            ik_joint_qd[li] = v
            state_qd[li] = v

        if tid < right_count:
            ri = right_indices[tid]
            lo_r = joint_limit_lower[ri]
            hi_r = joint_limit_upper[ri]
            q_end_r = lo_r + right_open * (hi_r - lo_r)
            q_target_frame[ri] = q_end_r

            alpha = float(substep_index + 1) / float(sim_substeps)
            target_r = q0_frame[ri] + alpha * (q_end_r - q0_frame[ri])
            v_r = (target_r - current_q[ri]) / sim_dt
            joint_target[ri] = v_r
            ik_joint_qd[ri] = v_r
            state_qd[ri] = v_r

    if tid < controllable_count:
        ci = controllable_indices[tid]
        alpha = float(substep_index + 1) / float(sim_substeps)
        target_c = q0_frame[ci] + alpha * (q_target_frame[ci] - q0_frame[ci])
        v_c = (target_c - current_q[ci]) / sim_dt
        joint_target[ci] = target_c
        ik_joint_qd[ci] = v_c
        state_qd[ci] = v_c



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

    INTERACTIVE_QUEUE = 2
    """Interactive control with gizmo, splits motion into 30 steps."""

    INTERACTIVE_NO_QUEUE = 3
    """Interactive control with gizmo, no queue (direct response with optimizations)."""

    QUEUE_NO_CLAMP = 4
    """Interactive control with gizmo and queue, but without displacement clamping."""

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


class Example:
    """Example class for lift2 robot manipulating cloth simulation.
    
    This class sets up a dual-arm robot (lift2) with cloth manipulation capabilities,
    including IK control, physics simulation, and interactive/trajectory-based animation.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Constants
    # ═══════════════════════════════════════════════════════════════════════════
    IK_ITERATIONS = 24
    
    # Joint control gains
    ARM_JOINT_KE = 3000.0
    ARM_JOINT_KD = 10.0
    GRIPPER_JOINT_KE = 3000.0
    GRIPPER_JOINT_KD = 10.0
    
    # Gripper limits
    GRIPPER_LIMIT_LOWER = 0.005  # Leave a small gap to avoid penetration
    GRIPPER_LIMIT_UPPER = 0.044
    
    # Robot pose
    ROBOT_BASE_HEIGHT = 0.17  # meters
    
    # Scene objects
    TABLE_POS = wp.vec3(1.0, 0.0, 0.201)
    TABLE_SIZE = (0.6, 0.6, 0.2)  # half extents
    BOX_POS = wp.vec3(0.6, 0.0, 0.43)
    BOX_SIZE = 0.03  # half extent
    BOX_DENSITY = 100.0
    
    # Cloth properties
    CLOTH_POS = wp.vec3(0.7, 0.0, 0.5)
    CLOTH_ROTATION_ANGLE = np.pi * 0.5
    CLOTH_DENSITY = 0.2
    CLOTH_SCALE = 0.01
    
    def __init__(self, viewer, args):
        """Initialize the simulation example.
        
        Args:
            viewer: Newton viewer instance for visualization
            args: Command-line arguments containing simulation parameters
        """
        # ═══════════════════════════════════════════════════════════════════════════
        # 1. Timing and Performance Tracking
        # ═══════════════════════════════════════════════════════════════════════════
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._substep_index = 0
        self.last_step_time = None
        self.fps_window_size = 30
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 2. Command-Line Arguments
        # ═══════════════════════════════════════════════════════════════════════════
        self.gripper_control_type = GripperControlType(args.gripper_control_type)
        self.animation_type = AnimationType(args.animation_type)
        self.use_mujoco_cpu = args.use_mujoco_cpu
        self.sim_vbd_iterations = args.vbd_iterations
        self.trajectory_queue_size = args.trajectory_queue_size
        self.queue_executing = False
        
        self.use_dump_image = False
        self.use_dump_joint = False
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 3. Physics Parameters
        # ═══════════════════════════════════════════════════════════════════════════
        self._init_physics_parameters()
        
        self.viewer = viewer

        # ═══════════════════════════════════════════════════════════════════════════
        # 4. Build Robot Model
        # ═══════════════════════════════════════════════════════════════════════════
        franka = self._build_robot_model()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 5. Identify Joint Groups and End Effectors
        # ═══════════════════════════════════════════════════════════════════════════
        self._identify_bodies_and_joints(franka)
        
        # Initialize joint recording if needed
        if self.use_dump_joint:
            self.joint_q_seq = np.empty((0, franka.joint_dof_count), dtype=np.float32)
            self.openness_seq = np.empty((0, 2), dtype=np.float32)

        # ═══════════════════════════════════════════════════════════════════════════
        # 6. Configure Joint Controllers
        # ═══════════════════════════════════════════════════════════════════════════
        self.robot_joint_q_cnt = len(franka.joint_q)
        self._configure_joint_controllers(franka)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 7. Add Scene Objects (Table, Box, Cloth)
        # ═══════════════════════════════════════════════════════════════════════════
        self._add_scene_objects(franka)
        
        franka.color()

        # ═══════════════════════════════════════════════════════════════════════════
        # 8. Finalize Model and Initialize Computational Components
        # ═══════════════════════════════════════════════════════════════════════════
        self.model = franka.finalize(requires_grad=False)
        
        # Apply cloth parameters to model
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        # ═══════════════════════════════════════════════════════════════════════════
        # 10. Initialize GPU Buffers and Compute Graphs
        # ═══════════════════════════════════════════════════════════════════════════
        self._init_gpu_buffers()
        
        # Warp compute graphs (initialized later in capture())
        self.ik_graph = None
        self.physics_graph = None

        # ═══════════════════════════════════════════════════════════════════════════
        # 11. Setup Viewer
        # ═══════════════════════════════════════════════════════════════════════════
        self._setup_viewer()

        # ═══════════════════════════════════════════════════════════════════════════
        # 12. Initialize Simulation States
        # ═══════════════════════════════════════════════════════════════════════════
        self.state = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()

        # ═══════════════════════════════════════════════════════════════════════════
        # 13. Initialize End Effector Control
        # ═══════════════════════════════════════════════════════════════════════════
        self._init_end_effector_control()

        # ═══════════════════════════════════════════════════════════════════════════
        # 14. Initialize Trajectory Queue (for INTERACTIVE_QUEUE mode)
        # ═══════════════════════════════════════════════════════════════════════════
        self._init_trajectory_queue()
        
        # Pre-allocate buffers for IK target updates
        self._ik_pos_buffer = np.zeros((1, 3), dtype=np.float32)
        self._ik_rot_buffer = np.zeros((1, 4), dtype=np.float32)

        # ═══════════════════════════════════════════════════════════════════════════
        # 15. Setup IK Objectives
        # ═══════════════════════════════════════════════════════════════════════════
        self._setup_ik_objectives()
        
        # Variables the IK solver will update
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.ik_joint_qd = wp.array(self.model.joint_qd, shape=(self.model.joint_dof_count))
        self.ik_iters = self.IK_ITERATIONS

        # Trajectory animation (for TRAJECTORY mode)
        self.trajectory_animation = KeyFrameTrajectoryAnimation()
        self.trajectory_animation.init_lift2_folding()

        # ═══════════════════════════════════════════════════════════════════════════
        # 16. Initialize Solvers
        # ═══════════════════════════════════════════════════════════════════════════
        self._init_solvers()
        
        # Capture CUDA graphs for better performance
        self.capture()

    # ═══════════════════════════════════════════════════════════════════════════
    # Helper Methods for Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_physics_parameters(self):
        """Initialize physics and material parameters."""
        # Contact parameters
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003
        self.soft_contact_ke = 500
        self.soft_contact_kd = 5e-3
        self.robot_friction = 1.5
        self.table_friction = 0.25
        self.self_contact_friction = 0.25

        # Elasticity parameters for cloth
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6
        self.bending_ke = 1e-4
        self.bending_kd = 1e-3
    
    def _build_robot_model(self):
        """Build the robot model with URDF and ground plane.
        
        Returns:
            franka: ModelBuilder instance with robot loaded
        """
        franka = newton.ModelBuilder()
        
        franka.add_urdf(
            newton.examples.get_asset("lift2_urdf/fixed_robot.urdf"),
            floating=False,
            enable_self_collisions=False,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.ROBOT_BASE_HEIGHT))
        )
        franka.add_ground_plane()
        
        return franka
    
    def _identify_bodies_and_joints(self, builder):
        """Identify end effectors and categorize joints into groups.
        
        Args:
            builder: ModelBuilder instance
        """
        print("=== Body Information ===")
        for i in range(builder.body_count):
            print(f"body {i}, key={builder.body_key[i]}")
            if builder.body_key[i] in left_ee_body_names:
                self.lee_index = i
                print(f"  >> left end-effector")
            if builder.body_key[i] in right_ee_body_names:
                self.ree_index = i
                print(f"  >> right end-effector")

        print("=== Joint Information ===")
        print(f"#joint_dof={builder.joint_dof_count}, #joint_coord={builder.joint_coord_count}")
        
        # Initialize joint group arrays
        self.fixed_joint_indices = np.array([], dtype=int)
        self.controllable_joint_indices = np.array([], dtype=int)
        self.left_gripper_joint_indices = np.array([], dtype=int)
        self.right_gripper_joint_indices = np.array([], dtype=int)

        # Categorize joints
        cnt = 0
        for i in range(builder.joint_count):
            dof_dim = builder.joint_dof_dim[i]
            print(f"joint {i}, key={builder.joint_key[i]}, type={builder.joint_type[i]}, "
                  f"link={builder.joint_parent[i]} -> {builder.joint_child[i]}, "
                  f"dof_dim={dof_dim}, dof_start {cnt}, "
                  f"dof_lim=[{builder.joint_limit_lower[cnt]}, {builder.joint_limit_upper[cnt]}]")

            dof_start = cnt
            dof_end = cnt + dof_dim[0] + dof_dim[1]
            
            # Categorize joint based on name
            for j in range(dof_start, dof_end):
                if builder.joint_key[i] in fixed_joint_names:
                    self.fixed_joint_indices = np.append(self.fixed_joint_indices, [j])
                elif builder.joint_key[i] in controllable_joint_names:
                    self.controllable_joint_indices = np.append(self.controllable_joint_indices, [j])
                elif builder.joint_key[i] in left_gripper_joint_names:
                    self.left_gripper_joint_indices = np.append(self.left_gripper_joint_indices, [j])
                elif builder.joint_key[i] in right_gripper_joint_names:
                    self.right_gripper_joint_indices = np.append(self.right_gripper_joint_indices, [j])

            cnt += dof_dim[0] + dof_dim[1]

        print(f"joint dq cnt check: {cnt} == {builder.joint_dof_count}")
        print("fixed joint", self.fixed_joint_indices)
        print("controllable joint", self.controllable_joint_indices)
        print("left joint", self.left_gripper_joint_indices)
        print("right joint", self.right_gripper_joint_indices)
    
    def _configure_joint_controllers(self, builder):
        """Configure control modes and gains for all joints.
        
        Args:
            builder: ModelBuilder instance
        """
        # Configure arm joints (target position control)
        for i in self.controllable_joint_indices:
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            builder.joint_target_ke[i] = self.ARM_JOINT_KE
            builder.joint_target_kd[i] = self.ARM_JOINT_KD

        # Disable control for fixed joints
        for i in self.fixed_joint_indices:
            builder.joint_dof_mode[i] = newton.JointMode.NONE
            builder.joint_limit_lower[i] = 0
            builder.joint_limit_upper[i] = 0

        # Configure gripper joints
        gripper_indices = np.concatenate((self.left_gripper_joint_indices, 
                                          self.right_gripper_joint_indices))
        for i in gripper_indices:
            builder.joint_limit_lower[i] = self.GRIPPER_LIMIT_LOWER
            builder.joint_limit_upper[i] = self.GRIPPER_LIMIT_UPPER

            if self.gripper_control_type == GripperControlType.NONE:
                builder.joint_dof_mode[i] = newton.JointMode.NONE
            elif self.gripper_control_type == GripperControlType.TARGET_POSITION:
                builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
                builder.joint_target_ke[i] = self.GRIPPER_JOINT_KE
                builder.joint_target_kd[i] = self.GRIPPER_JOINT_KD
            elif self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
                builder.joint_dof_mode[i] = newton.JointMode.TARGET_VELOCITY
                builder.joint_target_kd[i] = self.GRIPPER_JOINT_KD
    
    def _add_scene_objects(self, builder):
        """Add table, box, and cloth to the scene.
        
        Args:
            builder: ModelBuilder instance
        """
        # Add fixed table
        body_table = builder.add_body()
        builder.add_joint_fixed(-1, body_table)
        builder.add_shape_box(
            body_table,
            xform=wp.transform(p=self.TABLE_POS, q=wp.quat_identity()),
            hx=self.TABLE_SIZE[0],
            hy=self.TABLE_SIZE[1],
            hz=self.TABLE_SIZE[2]
        )

        # # Add movable box
        # body_box = builder.add_body(xform=wp.transform(p=self.BOX_POS, q=wp.quat_identity()))
        # builder.add_joint_free(body_box)
        # builder.add_shape_box(
        #     body_box,
        #     hx=self.BOX_SIZE,
        #     hy=self.BOX_SIZE,
        #     hz=self.BOX_SIZE,
        #     cfg=newton.ModelBuilder.ShapeConfig(density=self.BOX_DENSITY)
        # )

        # Set friction for all shapes
        for i in range(len(builder.shape_material_mu)):
            builder.shape_material_mu[i] = 1.0
            builder.shape_material_ka[i] = 0.002
            builder.shape_is_solid[i] = True

        # Load and add cloth mesh
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("PulseAsset/cloth/garment-tri.usdc"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/World/mesh/Mesh"))
        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        print("=== Cloth Information ===")
        print(f"vertices={mesh_points.shape}, faces={mesh_indices.shape}")

        vertices = [wp.vec3(v) for v in mesh_points]
        builder.add_cloth_mesh(
            vertices=vertices,
            indices=mesh_indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.CLOTH_ROTATION_ANGLE),
            pos=self.CLOTH_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=self.CLOTH_DENSITY,
            scale=self.CLOTH_SCALE,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )
    
    def _init_gpu_buffers(self):
        """Initialize GPU buffers for control and IK."""
        self.left_gripper_joint_indices_wp = wp.array(
            self.left_gripper_joint_indices, dtype=int, device=self.model.device
        )
        self.right_gripper_joint_indices_wp = wp.array(
            self.right_gripper_joint_indices, dtype=int, device=self.model.device
        )
        self.controllable_joint_indices_wp = wp.array(
            self.controllable_joint_indices, dtype=int, device=self.model.device
        )
        self.q0_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.q_target_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.gripper_params_wp = wp.zeros(4, dtype=float, device=self.model.device)
    
    def _setup_viewer(self):
        """Configure viewer settings."""
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            camera_pos = type(self.viewer.camera.pos)(3.0, 0, 1.4)
            self.viewer.camera.pos = camera_pos
            self.viewer.camera.pitch = -20
    
    def _init_end_effector_control(self):
        """Initialize end effector transforms and gripper states."""
        # Initialize gripper openness
        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0
        self.left_gripper_state = 1.0
        self.right_gripper_state = 1.0

        # Get initial end effector transforms
        body_q_np = self.state.body_q.numpy()
        initial_lee_tf = wp.transform(*body_q_np[self.lee_index])
        initial_ree_tf = wp.transform(*body_q_np[self.ree_index])
        
        # Statistics for debugging displacement limiting
        self.clamp_stats = {
            'pos_clamp_count': 0,
            'rot_clamp_count': 0,
            'total_frames': 0,
            'max_pos_distance': 0.0,
            'max_rot_angle': 0.0
        }
        
        # Create independent transform objects for gizmo control
        self.gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        
        # Previous frame transforms for displacement limiting
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        
        # Initial end effector transforms for IK objectives
        self.lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
    
    def _init_trajectory_queue(self):
        """Initialize trajectory queue for INTERACTIVE_QUEUE mode."""
        self.lee_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.ree_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.queue_index = 0
        
        # Cache targets to avoid GPU->CPU sync
        self.cached_lee_target_tf = self.lee_tf
        self.cached_ree_target_tf = self.ree_tf
        
        # Pre-compute interpolation alphas
        self.trajectory_alphas = np.linspace(
            1.0 / self.trajectory_queue_size,
            1.0,
            self.trajectory_queue_size,
            dtype=np.float32
        )
    
    def _setup_ik_objectives(self):
        """Setup IK objectives for dual-arm control."""
        total_residuals = 2 * 6 + self.model.joint_coord_count
        
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Left end effector position objective
        self.l_pos_obj = ik.IKPositionObjective(
            link_index=self.lee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.lee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Left end effector rotation objective
        self.l_rot_obj = ik.IKRotationObjective(
            link_index=self.lee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.lee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        # Right end effector position objective
        self.r_pos_obj = ik.IKPositionObjective(
            link_index=self.ree_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ree_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
        )

        # Right end effector rotation objective
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
    
    def _init_solvers(self):
        """Initialize IK, rigid body, and cloth solvers."""
        # IK solver
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.ik_joint_q,
            objectives=[
                self.l_pos_obj,
                self.l_rot_obj,
                self.r_pos_obj,
                self.r_rot_obj,
                self.obj_joint_limits
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.MIXED,
        )

        # Rigid body solver
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=150000,
            ncon_per_world=150000,
            solver='newton',
            cone="elliptic",
            use_mujoco_cpu=self.use_mujoco_cpu,
            use_mujoco_contacts=False,
            contact_stiffness_time_const=self.sim_dt
        )

        # Cloth solver
        self.model.edge_rest_angle.zero_()
        self.cloth_solver = newton.solvers.SolverVBD(
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Main Simulation Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def capture(self):
        """Capture IK and physics simulation into CUDA graphs for better performance."""
        self.ik_graph = None
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
            # Capture IK simulation
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
        """Run physics simulation for all substeps."""
        for s in range(self.sim_substeps):
            self._substep_index = s

            # Update controls for this substep on device
            current_q_flat = self.state_0.joint_q.reshape((self.model.joint_coord_count,))
            left_count = int(self.left_gripper_joint_indices_wp.shape[0])
            right_count = int(self.right_gripper_joint_indices_wp.shape[0])
            controllable_count = int(self.controllable_joint_indices_wp.shape[0])
            dim = max(max(left_count, right_count), controllable_count)
            wp.launch(
                kernel=_update_control_kernel,
                dim=dim,
                inputs=[
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.left_gripper_joint_indices_wp,
                    self.right_gripper_joint_indices_wp,
                    self.controllable_joint_indices_wp,
                    self.q0_frame_wp,
                    self.q_target_frame_wp,
                    current_q_flat,
                    self.control.joint_target,
                    self.ik_joint_qd,
                    self.state_0.joint_qd,
                    left_count,
                    right_count,
                    controllable_count,
                    self.sim_substeps,
                    int(self._substep_index),
                    self.sim_dt,
                    self.gripper_params_wp,
                    int(self.gripper_control_type),
                ],
                device=self.model.device,
            )

            # Collide and clear forces; filter gripper contacts if opened
            self.contacts = self.model.collide(self.state_0)
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Temporarily clear particle info for rigid solver
            particle_count = self.model.particle_count
            self.model.particle_count = 0

            # Rigid body step
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Recover particle info
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            # Cloth step
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # Swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def _interpolate_transform(self, tf_start, tf_end, alpha):
        """Interpolate between two transforms using alpha in [0, 1].
        
        Optimized for CPU execution with minimal overhead.
        Note: GPU kernel not used here because:
        - Only 30 iterations (too small for GPU efficiency)
        - CPU execution is already very fast (~0.05ms)
        - GPU launch + transfer overhead > computation time
        """
        # Interpolate position (linear) - 3 FLOPs
        pos_start = wp.transform_get_translation(tf_start)
        pos_end = wp.transform_get_translation(tf_end)
        pos_interp = pos_start + alpha * (pos_end - pos_start)
        
        # Interpolate rotation (slerp) - optimized quaternion interpolation
        rot_start = wp.transform_get_rotation(tf_start)
        rot_end = wp.transform_get_rotation(tf_end)
        rot_interp = wp.quat_slerp(rot_start, rot_end, alpha)
        
        return wp.transform(pos_interp, rot_interp)

    def _has_gizmo_moved(self, threshold_pos=0.001, threshold_rot=0.1):
        """Check if gizmo has moved significantly since last recorded position."""
        # Check left end effector
        lee_pos_curr = wp.transform_get_translation(self.gizmo_lee_tf)
        lee_pos_prev = wp.transform_get_translation(self.prev_gizmo_lee_tf)
        lee_pos_diff = wp.length(lee_pos_curr - lee_pos_prev)
        
        lee_rot_curr = wp.transform_get_rotation(self.gizmo_lee_tf)
        lee_rot_prev = wp.transform_get_rotation(self.prev_gizmo_lee_tf)
        lee_rot_dot = abs(wp.dot(lee_rot_curr, lee_rot_prev))
        lee_rot_diff = 2.0 * wp.acos(wp.clamp(lee_rot_dot, -1.0, 1.0))
        
        # Check right end effector
        ree_pos_curr = wp.transform_get_translation(self.gizmo_ree_tf)
        ree_pos_prev = wp.transform_get_translation(self.prev_gizmo_ree_tf)
        ree_pos_diff = wp.length(ree_pos_curr - ree_pos_prev)
        
        ree_rot_curr = wp.transform_get_rotation(self.gizmo_ree_tf)
        ree_rot_prev = wp.transform_get_rotation(self.prev_gizmo_ree_tf)
        ree_rot_dot = abs(wp.dot(ree_rot_curr, ree_rot_prev))
        ree_rot_diff = 2.0 * wp.acos(wp.clamp(ree_rot_dot, -1.0, 1.0))
        
        # Return True if any significant movement detected (only end effector, not gripper)
        return (lee_pos_diff > threshold_pos or lee_rot_diff > threshold_rot or
                ree_pos_diff > threshold_pos or ree_rot_diff > threshold_rot)

    def _push_targets_from_gizmos_queue_no_clamp(self):
        """Read gizmo-updated transform and create a trajectory queue without clamping.
        
        This method is similar to _push_targets_from_gizmos_queue but does not apply
        displacement or rotation limits, allowing larger movements per frame.
        
        Key differences from INTERACTIVE_QUEUE:
        - No clamping of displacement or rotation
        - Allows fast, long-distance movements
        - Still provides smooth interpolation via queue
        """
        # Handle gripper control via keyboard (immediate response, no queue)
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                self.open_left_gripper -= 0.05
            else:
                self.open_left_gripper += 0.05

            if self.viewer.is_key_down("2"):
                self.open_right_gripper -= 0.05
            else:
                self.open_right_gripper += 0.05
                
            self.open_left_gripper = np.clip(self.open_left_gripper, 0.0, 1.0)
            self.open_right_gripper = np.clip(self.open_right_gripper, 0.0, 1.0)
        
        # Check if gizmo has moved significantly
        gizmo_moved = self._has_gizmo_moved(threshold_pos=0.001)
        if gizmo_moved:
            print(f"Gizmo moved (no clamp mode): {gizmo_moved}!!!!")

        if gizmo_moved:
            # Use gizmo positions directly without clamping
            target_lee_tf = self.gizmo_lee_tf
            target_ree_tf = self.gizmo_ree_tf
            
            # Use cached target transforms to avoid GPU->CPU sync
            current_lee_tf = self.cached_lee_target_tf
            current_ree_tf = self.cached_ree_target_tf
            
            # Fill pre-allocated trajectory queues by interpolating from current to target
            # No clamping applied - full range movement allowed
            lee_pos_start = wp.transform_get_translation(current_lee_tf)
            lee_pos_end = wp.transform_get_translation(target_lee_tf)
            lee_rot_start = wp.transform_get_rotation(current_lee_tf)
            lee_rot_end = wp.transform_get_rotation(target_lee_tf)
            
            ree_pos_start = wp.transform_get_translation(current_ree_tf)
            ree_pos_end = wp.transform_get_translation(target_ree_tf)
            ree_rot_start = wp.transform_get_rotation(current_ree_tf)
            ree_rot_end = wp.transform_get_rotation(target_ree_tf)
            
            for i, alpha in enumerate(self.trajectory_alphas):
                # Left end effector interpolation (inlined for performance)
                alpha_f = float(alpha)
                lee_pos_interp = lee_pos_start + alpha_f * (lee_pos_end - lee_pos_start)
                lee_rot_interp = wp.quat_slerp(lee_rot_start, lee_rot_end, alpha_f)
                self.lee_tf_queue[i] = wp.transform(lee_pos_interp, lee_rot_interp)
                
                # Right end effector interpolation (inlined for performance)
                ree_pos_interp = ree_pos_start + alpha_f * (ree_pos_end - ree_pos_start)
                ree_rot_interp = wp.quat_slerp(ree_rot_start, ree_rot_end, alpha_f)
                self.ree_tf_queue[i] = wp.transform(ree_pos_interp, ree_rot_interp)
            
            # Start executing the new queue
            self.queue_executing = True
            self.queue_index = 0  # Restart from beginning
        
        # IMPORTANT: Always update previous transforms at the end of each frame
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_lee_tf),
            wp.transform_get_rotation(self.gizmo_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(self.gizmo_ree_tf),
            wp.transform_get_rotation(self.gizmo_ree_tf)
        )
        
        # Execute current step in the queue (same as INTERACTIVE_QUEUE)
        if self.queue_executing and self.queue_index < len(self.lee_tf_queue):
            # Get current target from queue
            target_lee_tf = self.lee_tf_queue[self.queue_index]
            target_ree_tf = self.ree_tf_queue[self.queue_index]
            
            # Direct GPU array assignment
            lee_pos = wp.transform_get_translation(target_lee_tf)
            lee_rot = wp.transform_get_rotation(target_lee_tf)
            ree_pos = wp.transform_get_translation(target_ree_tf)
            ree_rot = wp.transform_get_rotation(target_ree_tf)
            
            # Left end effector
            self._ik_pos_buffer[0, 0] = lee_pos[0]
            self._ik_pos_buffer[0, 1] = lee_pos[1]
            self._ik_pos_buffer[0, 2] = lee_pos[2]
            self.l_pos_obj.target_positions.assign(self._ik_pos_buffer)
            
            self._ik_rot_buffer[0, 0] = lee_rot[0]
            self._ik_rot_buffer[0, 1] = lee_rot[1]
            self._ik_rot_buffer[0, 2] = lee_rot[2]
            self._ik_rot_buffer[0, 3] = lee_rot[3]
            self.l_rot_obj.target_rotations.assign(self._ik_rot_buffer)
            
            # Right end effector
            self._ik_pos_buffer[0, 0] = ree_pos[0]
            self._ik_pos_buffer[0, 1] = ree_pos[1]
            self._ik_pos_buffer[0, 2] = ree_pos[2]
            self.r_pos_obj.target_positions.assign(self._ik_pos_buffer)
            
            self._ik_rot_buffer[0, 0] = ree_rot[0]
            self._ik_rot_buffer[0, 1] = ree_rot[1]
            self._ik_rot_buffer[0, 2] = ree_rot[2]
            self._ik_rot_buffer[0, 3] = ree_rot[3]
            self.r_rot_obj.target_rotations.assign(self._ik_rot_buffer)
            
            # Update cached targets
            self.cached_lee_target_tf = target_lee_tf
            self.cached_ree_target_tf = target_ree_tf
            
            # Update gizmo positions to match queue execution
            # self.gizmo_lee_tf = wp.transform(lee_pos, lee_rot)
            # self.gizmo_ree_tf = wp.transform(ree_pos, ree_rot)
            
            # Advance queue index
            self.queue_index += 1
            
            # Check if queue is finished
            if self.queue_index >= len(self.lee_tf_queue):
                self.queue_executing = False
                self.queue_index = 0


    def step(self):
        """Execute one simulation step."""
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.left_gripper_state = self.open_left_gripper
            self.right_gripper_state = self.open_right_gripper

        if self.animation_type == AnimationType.INTERACTIVE:
            self._push_targets_from_gizmos()

        if self.animation_type == AnimationType.QUEUE_NO_CLAMP:
            self._push_targets_from_gizmos_queue_no_clamp()


        # IK step, update self.ik_joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        ik_joint_q = self.ik_joint_q.flatten()
        
        # Copy data directly on GPU to avoid CPU-GPU transfer
        # Use warp.copy for efficient GPU-to-GPU copy
        state_q_flat = self.state_0.joint_q.reshape((self.model.joint_coord_count,))
        wp.copy(self.q0_frame_wp, state_q_flat)
        wp.copy(self.q_target_frame_wp, ik_joint_q)
        
        # Update gripper params on host then transfer once (small data, acceptable overhead)
        # [left_prev, left_target, right_prev, right_target]
        # Note: This is a small 4-element array, the overhead is minimal
        gripper_params_host = np.array([
            float(self.left_gripper_state),
            float(self.open_left_gripper),
            float(self.right_gripper_state),
            float(self.open_right_gripper),
        ], dtype=np.float32)
        self.gripper_params_wp.assign(gripper_params_host)

        # Physics step for all substeps (loop is inside physics_simulate for CUDA graph)
        if self.physics_graph:
            # print("Launching physics simulation from CUDA graph...")
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()
            

        self.sim_time += self.frame_dt
        self.sim_frame += 1
        
        # Calculate and print FPS
        current_time = time.time()
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            current_fps = 1.0 / step_duration if step_duration > 0 else 0            
            # Print FPS info (skip frequent printing in GPU-optimized interactive modes for better performance)
            if self.animation_type not in (AnimationType.INTERACTIVE_QUEUE, AnimationType.INTERACTIVE_NO_QUEUE, AnimationType.QUEUE_NO_CLAMP) or self.sim_frame % 30 == 0:
                print(f'Frame {self.sim_frame}: FPS = {current_fps:.2f}, Step time = {step_duration*1000:.2f}ms')
        
        self.last_step_time = current_time
        
        if self.use_dump_joint:
            joint_q_np = self.state_0.joint_q.numpy()
            self.joint_q_seq = np.vstack((self.joint_q_seq, joint_q_np[0:self.robot_joint_q_cnt]))
            self.openness_seq = np.vstack((self.openness_seq, np.array([self.open_left_gripper, self.open_right_gripper])))

            if self.sim_frame == 32 * self.fps:
                np.savez('lift2_manipulating_cloth.npz', joint_q=self.joint_q_seq, openness=self.openness_seq)
        
        self.left_gripper_state = self.open_left_gripper
        self.right_gripper_state = self.open_right_gripper

    def render(self):
        """Render the current frame."""
        self.viewer.begin_frame(self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            # Register gizmo (viewer will draw & mutate transform in-place)
            # Use gizmo values directly (they represent user input, not interpolated values)
            # The gizmo transforms are mutated in-place by viewer, so we need to pass the
            # current gizmo values (which will be updated by user interaction)
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        
        if self.animation_type == AnimationType.INTERACTIVE_QUEUE:
            # Register gizmo for INTERACTIVE_QUEUE mode
            # Always accept gizmo input, will create new trajectory when moved
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        
        if self.animation_type == AnimationType.INTERACTIVE_NO_QUEUE:
            # Register gizmo for INTERACTIVE_NO_QUEUE mode
            # Direct gizmo control with optimizations
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        
        if self.animation_type == AnimationType.QUEUE_NO_CLAMP:
            # Register gizmo for QUEUE_NO_CLAMP mode
            # Queue-based control without displacement clamping
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        
        self.viewer.log_state(self.state_0)

        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        # Only synchronize for non-GPU-optimized modes
        # For INTERACTIVE_QUEUE, INTERACTIVE_NO_QUEUE, and QUEUE_NO_CLAMP, rely on implicit synchronization at render time
        if self.animation_type not in (AnimationType.INTERACTIVE_QUEUE, AnimationType.INTERACTIVE_NO_QUEUE, AnimationType.QUEUE_NO_CLAMP):
            wp.synchronize()

        if self.use_dump_image:
            io_util.dump_gl_frame_image(
                self.viewer.renderer._screen_width,
                self.viewer.renderer._screen_height,
                f"img_{self.sim_frame}.png"
            )

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    if not os.environ.get("DISPLAY"):
        parser.set_defaults(viewer="null", headless=True)
    
    # Add custom arguments for control configuration
    parser.add_argument("--gripper-control-type", type=int, default=1,
                        choices=[0, 1, 2],
                        help="Gripper control: 0=NONE, 1=TARGET_POSITION, 2=TARGET_VELOCITY")
    
    parser.add_argument("--animation-type", type=int, default=4,
                        choices=[0, 1, 2, 3, 4],
                        help="Animation: 0=INTERACTIVE, 1=TRAJECTORY, 2=INTERACTIVE_QUEUE, 3=INTERACTIVE_NO_QUEUE, 4=QUEUE_NO_CLAMP")
    
    parser.add_argument("--vbd-iterations", type=int, default=7,
                        help="VBD iterations for cloth simulation")
    
    parser.add_argument("--use-mujoco-cpu", action="store_true", default=False,
                        help="Use MuJoCo CPU solver instead of GPU")
    
    parser.add_argument("--trajectory-queue-size", type=int, default=30,
                        help="Trajectory queue size for INTERACTIVE_QUEUE mode")
    
    # Simulation timing parameters
    parser.add_argument("--fps", type=int, default=30,
                        help="Simulation FPS (frames per second)")
    
    parser.add_argument("--sim-substeps", type=int, default=20,
                        help="Simulation substeps per frame")
    
    # Motion limits for gizmo control
    parser.add_argument("--max-displacement-per-frame", type=float, default=0.1,
                        help="Maximum displacement per frame in meters (e.g., 0.01 = 10mm)")
    
    parser.add_argument("--max-rotation-per-frame", type=float, default=0.1,
                        help="Maximum rotation per frame in radians (e.g., 0.1 ≈ 5.7°)")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
