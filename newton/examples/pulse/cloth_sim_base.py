"""
Base class for cloth manipulation simulation examples.

This module provides a base class that encapsulates common functionality
for cloth manipulation simulations, including:
- Physics configuration management
- Robot model building
- Joint and body identification
- Contact filtering setup
- GPU buffer initialization
- Common utility methods
"""

import warp as wp
import numpy as np
import json
import time
import newton
import newton.examples
import newton.ik as ik

from cloth_sim_types import (
    GripperControlType,
    fixed_joint_names,
    controllable_joint_names,
    left_ee_body_names,
    left_gripper_joint_names,
    right_ee_body_names,
    right_gripper_joint_names,
)


class ExampleBase:
    """
    Base class for cloth manipulation examples.
    
    This class provides common functionality that can be inherited by specific
    simulation examples. Subclasses should override specific methods to customize
    behavior while reusing the common infrastructure.
    
    Common functionality includes:
    - Physics parameter loading/saving
    - Robot model building
    - Joint categorization and configuration
    - Contact filtering setup
    - GPU buffer management
    - Utility methods for transforms and interpolation
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Physics Configuration Management
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _load_physics_config(self, config_file):
        """Load physics parameters from a JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Dictionary of parameter values
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Successfully loaded configuration from: {config_file}")
            return config.get('physics_parameters', {})
        except FileNotFoundError:
            print(f"⚠ Warning: Configuration file not found: {config_file}")
            print(f"  Using default parameters instead.")
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Failed to parse JSON in {config_file}: {e}")
            print(f"  Using default parameters instead.")
            return {}
        except Exception as e:
            print(f"⚠ Warning: Error loading configuration file: {e}")
            print(f"  Using default parameters instead.")
            return {}
    
    def save_physics_config(self, config_file):
        """Save current physics parameters to a JSON configuration file.
        
        This method should be overridden if additional parameters need to be saved.
        
        Args:
            config_file: Path to save the JSON configuration file
        """
        params = {
            'physics_parameters': {
                # Contact parameters
                'cloth_particle_radius': float(self.cloth_particle_radius),
                'cloth_body_contact_margin': float(self.cloth_body_contact_margin),
                'self_contact_radius': float(self.self_contact_radius),
                'self_contact_margin': float(self.self_contact_margin),
                'soft_contact_ke': float(self.soft_contact_ke),
                'soft_contact_kd': float(self.soft_contact_kd),
                'robot_friction': float(self.robot_friction),
                'table_friction': float(self.table_friction),
                'self_contact_friction': float(self.self_contact_friction),
                # Elasticity parameters for cloth
                'tri_ke': float(self.tri_ke),
                'tri_ka': float(self.tri_ka),
                'tri_kd': float(self.tri_kd),
                'bending_ke': float(self.bending_ke),
                'bending_kd': float(self.bending_kd),
            },
            'metadata': {
                'description': 'Physics parameters for cloth simulation',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fps': self.fps,
                'sim_substeps': self.sim_substeps,
                'vbd_iterations': self.sim_vbd_iterations,
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"✓ Successfully saved configuration to: {config_file}")
        except Exception as e:
            print(f"⚠ Error: Failed to save configuration file: {e}")
    
    def _init_physics_parameters(self, config_params=None):
        """Initialize physics and material parameters.
        
        Can be overridden by subclasses to use different default values.
        
        Args:
            config_params: Optional dictionary of parameters loaded from config file
        """
        # Default parameters tuned for cotton fabric
        defaults = {
            # Contact parameters
            'cloth_particle_radius': 0.008,
            'cloth_body_contact_margin': 0.01,
            'self_contact_radius': 0.001,
            'self_contact_margin': 0.002,
            'soft_contact_ke': 1000.0,
            'soft_contact_kd': 5e-3,
            'robot_friction': 1.5,
            'table_friction': 0.25,
            'self_contact_friction': 0.8,  # Cotton: moderate friction
            # Elasticity parameters for cotton cloth
            'tri_ke': 200.0,      # Cotton: medium stretch stiffness
            'tri_ka': 200.0,      # Cotton: medium area preservation
            'tri_kd': 5e-6,       # Cotton: moderate damping
            'bending_ke': 8e-2,   # Cotton: moderate bending resistance
            'bending_kd': 5e-1,   # Cotton: moderate bending damping
        }
        
        # Override defaults with config file values if provided
        if config_params:
            print(f"📝 Loading physics parameters from config file:")
            for key, value in config_params.items():
                if key in defaults:
                    defaults[key] = value
                    print(f"  {key}: {value}")
        
        # Set all parameters as instance attributes
        for key, value in defaults.items():
            setattr(self, key, value)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Robot Model Building
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _build_robot_model(self):
        """Build the robot model with URDF and ground plane.
        
        Can be overridden to use a different robot or configuration.
        
        Returns:
            ModelBuilder instance with robot loaded
        """
        builder = newton.ModelBuilder()
        
        builder.add_urdf(
            newton.examples.get_asset("lift2_urdf/fixed_robot.urdf"),
            floating=False,
            enable_self_collisions=False,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.ROBOT_BASE_HEIGHT))
        )
        
        # Add ground plane with explicit collision configuration
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            has_particle_collision=True,  # Enable collision with cloth
            has_shape_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        
        return builder
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Joint and Body Identification
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _identify_bodies_and_joints(self, builder):
        """Identify end effectors and categorize joints into groups.
        
        Can be extended by subclasses to identify additional bodies or joints.
        
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
        
        Can be overridden to use different control strategies.
        
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Contact Filtering Setup
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _setup_contact_filtering(self):
        """Prepare data structures for dynamic collision filtering.
        
        Collects shape IDs for grippers and end-effectors.
        All collision control is performed dynamically at runtime based on gripper state.
        """
        joint_child_np = self.model.joint_child.numpy()
        COLLIDE_PARTICLES = 1 << 2  # ShapeFlags.COLLIDE_PARTICLES
        
        def _collect_shape_ids_from_joints(joint_name_set):
            """Collect shape IDs corresponding to specified joint names."""
            bodies = [int(joint_child_np[i]) for i, key in enumerate(self.model.joint_key) if key in joint_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids
        
        def _collect_shape_ids_from_bodies(body_name_set):
            """Collect shape IDs corresponding to specified body names."""
            bodies = [i for i, key in enumerate(self.model.body_key) if key in body_name_set]
            shape_ids = set()
            for body_id in bodies:
                if body_id in self.model.body_shapes:
                    shape_ids.update(int(shape_id) for shape_id in self.model.body_shapes[body_id])
            return shape_ids

        # Collect shapes for left and right grippers (including end-effectors)
        left_gripper_shapes = _collect_shape_ids_from_joints(left_gripper_joint_names)
        left_ee_shapes = _collect_shape_ids_from_bodies(left_ee_body_names)
        left_shapes_combined = left_gripper_shapes | left_ee_shapes  # Union of sets
        
        right_gripper_shapes = _collect_shape_ids_from_joints(right_gripper_joint_names)
        right_ee_shapes = _collect_shape_ids_from_bodies(right_ee_body_names)
        right_shapes_combined = right_gripper_shapes | right_ee_shapes  # Union of sets

        # Collect shapes for controllable joints (arm joints excluding grippers)
        controllable_shapes = _collect_shape_ids_from_joints(controllable_joint_names)

        # Convert to numpy arrays
        if left_shapes_combined:
            self.left_gripper_shape_ids = np.array(sorted(left_shapes_combined), dtype=np.int32)
        else:
            self.left_gripper_shape_ids = np.zeros((0,), dtype=np.int32)
            
        if right_shapes_combined:
            self.right_gripper_shape_ids = np.array(sorted(right_shapes_combined), dtype=np.int32)
        else:
            self.right_gripper_shape_ids = np.zeros((0,), dtype=np.int32)
        
        if controllable_shapes:
            self.controllable_joint_shape_ids = np.array(sorted(controllable_shapes), dtype=np.int32)
        else:
            self.controllable_joint_shape_ids = np.zeros((0,), dtype=np.int32)
        
        self.COLLIDE_PARTICLES = COLLIDE_PARTICLES
        
        print(f"\n=== Contact Filtering Setup Complete ===")
        print(f"Left gripper shapes (including fl_link6): {len(self.left_gripper_shape_ids)}, IDs: {self.left_gripper_shape_ids.tolist()}")
        print(f"  - from gripper joints: {sorted(left_gripper_shapes)}")
        print(f"  - from fl_link6 body: {sorted(left_ee_shapes)}")
        print(f"Right gripper shapes (including fr_link6): {len(self.right_gripper_shape_ids)}, IDs: {self.right_gripper_shape_ids.tolist()}")
        print(f"  - from gripper joints: {sorted(right_gripper_shapes)}")
        print(f"  - from fr_link6 body: {sorted(right_ee_shapes)}")
        print(f"Controllable joint shapes: {len(self.controllable_joint_shape_ids)}, IDs: {self.controllable_joint_shape_ids.tolist()}")
        print(f"Dynamic collision filtering: disable gripper and end-effector collision during gripper release")
    
    def _disable_controllable_cloth_collision(self):
        """Disable collision between controllable joints and cloth particles at initialization.
        
        This method permanently disables collision between the robot arm joints 
        (specified in controllable_joint_names) and cloth particles.
        This is useful to prevent unwanted cloth-arm collisions during manipulation.
        
        Similar to _update_gripper_collision_filtering_cpu but applied once at init time.
        """
        # Get shape_flags numpy view
        shape_flags_np = self.model.shape_flags.numpy()
        
        # Disable particle collision for all controllable joint shapes
        for shape_id in self.controllable_joint_shape_ids:
            # Clear COLLIDE_PARTICLES flag (bit 2)
            shape_flags_np[shape_id] = shape_flags_np[shape_id] & ~self.COLLIDE_PARTICLES
        
        # Sync modified flags back to GPU (if on GPU)
        self.model.shape_flags.assign(shape_flags_np)
        
        print(f"\n=== Controllable Joint Collision Filtering ===")
        print(f"Disabled cloth collision for {len(self.controllable_joint_shape_ids)} controllable joint shapes")
        print(f"Shape IDs: {self.controllable_joint_shape_ids.tolist()}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GPU Buffer Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_gpu_buffers(self):
        """Initialize GPU buffers for control and IK.
        
        Can be extended by subclasses to initialize additional buffers.
        """
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
        
        # GPU buffers for collision filtering
        self.left_gripper_shape_ids_wp = wp.array(
            self.left_gripper_shape_ids, dtype=int, device=self.model.device
        )
        self.right_gripper_shape_ids_wp = wp.array(
            self.right_gripper_shape_ids, dtype=int, device=self.model.device
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _interpolate_transform(self, tf_start, tf_end, alpha):
        """Interpolate between two transforms using alpha in [0, 1].
        
        Uses linear interpolation for position and SLERP for rotation.
        
        Args:
            tf_start: Starting transform (wp.transform)
            tf_end: Ending transform (wp.transform)
            alpha: Interpolation factor (0=start, 1=end)
            
        Returns:
            Interpolated transform
        """
        # Interpolate position (linear)
        pos_start = wp.transform_get_translation(tf_start)
        pos_end = wp.transform_get_translation(tf_end)
        pos_interp = pos_start + alpha * (pos_end - pos_start)
        
        # Interpolate rotation (SLERP)
        rot_start = wp.transform_get_rotation(tf_start)
        rot_end = wp.transform_get_rotation(tf_end)
        rot_interp = wp.quat_slerp(rot_start, rot_end, alpha)
        
        return wp.transform(pos_interp, rot_interp)

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

    def _update_gripper_collision_filtering(self):
        """动态更新gripper碰撞过滤：只在夹爪松开过程中禁用碰撞（GPU版本）。
        
        规则：
        - 当gripper_state < open_gripper（正在松开/打开）时，禁用夹爪碰撞
        - 当gripper_state >= open_gripper（已关闭或保持）时，启用夹爪碰撞
        - 机器人非gripper部分（手臂等）不受影响，保持默认碰撞状态
        
        这样做的目的是让夹爪松开时能够穿过衣物，避免碰撞阻碍松开动作。
        
        注意：此版本使用GPU kernel，兼容CUDA graph capture。
        """
        # 判断gripper状态（state < target 表示正在松开）
        # 使用当前帧开始时的gripper state和目标值来判断
        left_opening = 1 if float(self.left_gripper_state) < float(self.open_left_gripper) else 0
        # left_opening = 1
        right_opening = 1 if float(self.right_gripper_state) < float(self.open_right_gripper) else 0
        
        # 计算kernel launch维度
        left_count = int(self.left_gripper_shape_ids_wp.shape[0])
        right_count = int(self.right_gripper_shape_ids_wp.shape[0])
        dim = max(left_count, right_count)
        
        if dim > 0:
            # 在GPU上更新collision flags
            wp.launch(
                kernel=_update_gripper_collision_kernel,
                dim=dim,
                inputs=[
                    self.model.shape_flags,
                    self.left_gripper_shape_ids_wp,
                    self.right_gripper_shape_ids_wp,
                    left_count,
                    right_count,
                    left_opening,
                    right_opening,
                    self.COLLIDE_PARTICLES,
                ],
                device=self.model.device,
            )

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
