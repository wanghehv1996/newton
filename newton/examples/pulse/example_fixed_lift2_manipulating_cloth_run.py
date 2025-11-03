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

@wp.kernel
def _filter_rigid_contacts_kernel(
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    disabled_shapes: wp.array(dtype=int),
    disabled_shape_count: wp.array(dtype=int),
    keep_mask: wp.array(dtype=int),
):
    """Mark contacts to keep (1) or remove (0) based on disabled shapes."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return
    
    shape0 = contact_shape0[tid]
    shape1 = contact_shape1[tid]
    
    # Check if either shape is disabled
    # Declare keep as a dynamic variable (required for break statement in Warp)
    keep = int(1)
    disabled_count = disabled_shape_count[0]
    for i in range(disabled_count):
        if shape0 == disabled_shapes[i] or shape1 == disabled_shapes[i]:
            keep = int(0)
            break
    
    keep_mask[tid] = keep

@wp.kernel
def _compact_rigid_contacts_kernel(
    contact_count: wp.array(dtype=int),
    keep_mask: wp.array(dtype=int),
    prefix_sum: wp.array(dtype=int),
    # Input arrays
    contact_shape0_in: wp.array(dtype=int),
    contact_shape1_in: wp.array(dtype=int),
    contact_point0_in: wp.array(dtype=wp.vec3),
    contact_point1_in: wp.array(dtype=wp.vec3),
    contact_normal_in: wp.array(dtype=wp.vec3),
    contact_thickness0_in: wp.array(dtype=float),
    contact_thickness1_in: wp.array(dtype=float),
    # Output arrays
    contact_shape0_out: wp.array(dtype=int),
    contact_shape1_out: wp.array(dtype=int),
    contact_point0_out: wp.array(dtype=wp.vec3),
    contact_point1_out: wp.array(dtype=wp.vec3),
    contact_normal_out: wp.array(dtype=wp.vec3),
    contact_thickness0_out: wp.array(dtype=float),
    contact_thickness1_out: wp.array(dtype=float),
):
    """Compact rigid contacts array using keep_mask."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return
    
    if keep_mask[tid] == 1:
        new_idx = prefix_sum[tid] - 1  # prefix_sum is 1-indexed
        contact_shape0_out[new_idx] = contact_shape0_in[tid]
        contact_shape1_out[new_idx] = contact_shape1_in[tid]
        contact_point0_out[new_idx] = contact_point0_in[tid]
        contact_point1_out[new_idx] = contact_point1_in[tid]
        contact_normal_out[new_idx] = contact_normal_in[tid]
        contact_thickness0_out[new_idx] = contact_thickness0_in[tid]
        contact_thickness1_out[new_idx] = contact_thickness1_in[tid]

@wp.kernel
def _get_new_count_kernel(
    prefix_sum: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    new_count: wp.array(dtype=int),
):
    """Get new contact count from prefix sum."""
    count = contact_count[0]
    if count > 0:
        new_count[0] = prefix_sum[count - 1]
    else:
        new_count[0] = 0

@wp.kernel
def _filter_soft_contacts_kernel(
    contact_count: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    disabled_shapes: wp.array(dtype=int),
    disabled_shape_count: wp.array(dtype=int),
    keep_mask: wp.array(dtype=int),
):
    """Mark soft contacts to keep (1) or remove (0) based on disabled shapes."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return
    
    shape = contact_shape[tid]
    
    # Check if shape is disabled
    # Declare keep as a dynamic variable (required for break statement in Warp)
    keep = int(1)
    disabled_count = disabled_shape_count[0]
    for i in range(disabled_count):
        if shape == disabled_shapes[i]:
            keep = int(0)
            break
    
    keep_mask[tid] = keep

@wp.kernel
def _compact_soft_contacts_kernel(
    contact_count: wp.array(dtype=int),
    keep_mask: wp.array(dtype=int),
    prefix_sum: wp.array(dtype=int),
    # Input arrays
    contact_particle_in: wp.array(dtype=int),
    contact_shape_in: wp.array(dtype=int),
    contact_body_pos_in: wp.array(dtype=wp.vec3),
    contact_body_vel_in: wp.array(dtype=wp.vec3),
    contact_normal_in: wp.array(dtype=wp.vec3),
    # Output arrays
    contact_particle_out: wp.array(dtype=int),
    contact_shape_out: wp.array(dtype=int),
    contact_body_pos_out: wp.array(dtype=wp.vec3),
    contact_body_vel_out: wp.array(dtype=wp.vec3),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """Compact soft contacts array using keep_mask."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return
    
    if keep_mask[tid] == 1:
        new_idx = prefix_sum[tid] - 1  # prefix_sum is 1-indexed
        contact_particle_out[new_idx] = contact_particle_in[tid]
        contact_shape_out[new_idx] = contact_shape_in[tid]
        contact_body_pos_out[new_idx] = contact_body_pos_in[tid]
        contact_body_vel_out[new_idx] = contact_body_vel_in[tid]
        contact_normal_out[new_idx] = contact_normal_in[tid]

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

def clamp_transform_delta(current_tf, target_tf, max_translation, max_rotation):
    """
    Clamp the transform delta between current and target transforms.
    Limits both displacement and rotation.
    
    Args:
        current_tf: Current transform (wp.transform)
        target_tf: Target transform (wp.transform)
        max_translation: Maximum allowed translation distance (meters)
        max_rotation: Maximum allowed rotation angle (radians)
    
    Returns:
        Clamped target transform (wp.transform)
    """
    # Get current and target positions
    current_pos = wp.transform_get_translation(current_tf)
    target_pos = wp.transform_get_translation(target_tf)
    
    # Calculate position delta
    pos_delta = target_pos - current_pos
    pos_distance = wp.length(pos_delta)
    
    # Clamp position if needed
    if pos_distance > max_translation:
        clamped_pos = current_pos + (pos_delta / pos_distance) * max_translation
    else:
        clamped_pos = target_pos
    
    # Get current and target rotations
    current_rot = wp.transform_get_rotation(current_tf)
    target_rot = wp.transform_get_rotation(target_tf)
    
    # Calculate rotation delta using quaternion
    # q_delta = q_target * q_current^-1
    current_rot_inv = wp.quat_inverse(current_rot)
    rot_delta = wp.mul(target_rot, current_rot_inv)
    
    # Get rotation angle from quaternion
    # For quaternion (w, x, y, z), angle = 2 * acos(w)
    rot_angle = 2.0 * wp.acos(wp.clamp(rot_delta[3], -1.0, 1.0))  # w is at index 3
    
    # Clamp rotation if needed
    if rot_angle > max_rotation:
        # Interpolate between current and target rotation
        t = max_rotation / rot_angle
        clamped_rot = wp.quat_slerp(current_rot, target_rot, t)
    else:
        clamped_rot = target_rot
    
    return wp.transform(clamped_pos, clamped_rot)

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

    INTERACTIVE_NEW = 2
    """Interactive control with gizmo, splits motion into 30 steps."""

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
    def __init__(self, viewer):
        # frame timing
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._substep_index = 0
        
        # FPS tracking
        self.last_step_time = None
        self.fps_window_size = 30  # Average over 30 frames
        

        self.gripper_control_type = GripperControlType.TARGET_POSITION
        self.use_mujoco_cpu = False
        # Animation type options:
        # - AnimationType.INTERACTIVE: Direct gizmo control (continuous)
        # - AnimationType.INTERACTIVE_NEW: Gizmo control with 30-step interpolation (queued execution)
        # - AnimationType.TRAJECTORY: Pre-recorded trajectory playback
        self.animation_type = AnimationType.INTERACTIVE_NEW
        self.use_dump_image = False
        self.use_dump_joint = False

        # VBD parameters
        if self.animation_type == AnimationType.INTERACTIVE:
            self.sim_vbd_iterations = 3    
        elif self.animation_type == AnimationType.TRAJECTORY:
            self.sim_vbd_iterations = 10
        elif self.animation_type == AnimationType.INTERACTIVE_NEW:
            self.sim_vbd_iterations = 6
        else:
            raise ValueError(f"Invalid animation type: {self.animation_type}")
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

        # Elasticity parameters
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
            franka.joint_target_ke[i] = 3000.0200
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
            franka.joint_limit_upper[i] = 0.044

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

        # Add the T-shirt (garment can be downloaded from https://gitee.pjlab.org.cn/L2/wanghui1/PulseAsset.git)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("PulseAsset/cloth/garment-tri.usdc"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/World/mesh/Mesh"))
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

        # Precompute shape indices that belong to gripper finger bodies for contact filtering
        self.left_gripper_body_indices = []
        self.right_gripper_body_indices = []
        joint_child_np = self.model.joint_child.numpy()
        for j_idx, j_key in enumerate(self.model.joint_key):
            if j_key in left_gripper_joint_names:
                self.left_gripper_body_indices.append(int(joint_child_np[j_idx]))
            if j_key in right_gripper_joint_names:
                self.right_gripper_body_indices.append(int(joint_child_np[j_idx]))

        def _collect_shapes(body_indices: list[int]) -> set[int]:
            shape_set: set[int] = set()
            for b in body_indices:
                b_idx = int(b)
                if b_idx in self.model.body_shapes:
                    for s in self.model.body_shapes[b_idx]:
                        shape_set.add(int(s))
            return shape_set

        self.left_gripper_shape_set = _collect_shapes(self.left_gripper_body_indices)
        self.right_gripper_shape_set = _collect_shapes(self.right_gripper_body_indices)

        # GPU arrays for contact filtering (CUDA graph compatible)
        max_disabled_shapes = len(self.left_gripper_shape_set) + len(self.right_gripper_shape_set)
        self.disabled_shapes_wp = wp.zeros(max_disabled_shapes, dtype=int, device=self.model.device)
        self.disabled_shape_count_wp = wp.zeros(1, dtype=int, device=self.model.device)
        
        # Pre-compute all possible disabled shape configurations for performance
        # This avoids rebuilding arrays every frame
        left_shapes = sorted(list(self.left_gripper_shape_set))
        right_shapes = sorted(list(self.right_gripper_shape_set))
        both_shapes = left_shapes + right_shapes
        
        self._disabled_shapes_cache = {
            (False, False): (np.array([], dtype=np.int32), 0),  # Neither gripper opening
            (True, False): (np.array(left_shapes, dtype=np.int32), len(left_shapes)),  # Left only
            (False, True): (np.array(right_shapes, dtype=np.int32), len(right_shapes)),  # Right only
            (True, True): (np.array(both_shapes, dtype=np.int32), len(both_shapes)),  # Both opening
        }
        
        # Track previous state to avoid redundant updates
        self._prev_disabled_state = None
        
        # Temporary buffers for filtering (reused each frame)
        # soft_contact_max is not stored in Model, compute it as shape_count * particle_count
        # This is the standard calculation used by collision pipelines
        soft_contact_max = self.model.shape_count * self.model.particle_count
        max_contacts = max(self.model.rigid_contact_max, soft_contact_max)
        self.keep_mask_wp = wp.zeros(max_contacts, dtype=int, device=self.model.device)
        self.prefix_sum_wp = wp.zeros(max_contacts, dtype=int, device=self.model.device)
        self.new_count_wp = wp.zeros(1, dtype=int, device=self.model.device)
        
        # Store soft_contact_max for later use in filtering kernels
        self.soft_contact_max = soft_contact_max

        # Set cloth parameter
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        # Warp compute graphs
        self.ik_graph = None
        self.physics_graph = None

        # Device index buffers for grippers (for GPU kernels)
        self.left_gripper_joint_indices_wp = wp.array(self.left_gripper_joint_indices, dtype=int, device=self.model.device)
        self.right_gripper_joint_indices_wp = wp.array(self.right_gripper_joint_indices, dtype=int, device=self.model.device)
        self.controllable_joint_indices_wp = wp.array(self.controllable_joint_indices, dtype=int, device=self.model.device)
        self.q0_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.q_target_frame_wp = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        self.gripper_params_wp = wp.zeros(4, dtype=float, device=self.model.device)

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
        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0
        self.left_gripper_state = 1.0
        self.right_gripper_state = 1.0

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state.body_q.numpy()
        initial_lee_tf = wp.transform(*body_q_np[self.lee_index])
        initial_ree_tf = wp.transform(*body_q_np[self.ree_index])
        
        # Interactive mode: instruction queue for smooth interpolation
        # Each instruction executes over 0.5 seconds (30 frames at 60 FPS)
        self.instruction_duration = 0.5  # seconds
        self.instruction_frames = int(self.instruction_duration * self.fps)  # 30 frames at 60 FPS
        self.current_instruction_start_time = 0.0
        # Maximum displacement per frame to limit speed (for direct gizmo control)
        self.max_displacement_per_frame = 0.01  # 5cm per frame at 60 FPS = 3 m/s max speed
        self.max_rotation_per_frame = 0.01  # radians per frame (~28 degrees)
        # Store gizmo input values (updated by viewer in render())
        # IMPORTANT: Create separate transform objects for gizmo to ensure they are independently mutable
        # Use slice assignment to copy the transform values
        self.gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        # Store previous frame transforms for displacement limiting
        self.prev_gizmo_lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.prev_gizmo_ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        # Store current interpolated end effector transforms
        self.lee_tf = wp.transform(
            wp.transform_get_translation(initial_lee_tf),
            wp.transform_get_rotation(initial_lee_tf)
        )
        self.ree_tf = wp.transform(
            wp.transform_get_translation(initial_ree_tf),
            wp.transform_get_rotation(initial_ree_tf)
        )
        # Store target states (from gizmo or keyboard, may be clamped)
        self.target_lee_tf = self.lee_tf  # Target left end effector transform
        self.target_ree_tf = self.ree_tf  # Target right end effector transform
        self.target_left_gripper = 1.0    # Target left gripper openness
        self.target_right_gripper = 1.0   # Target right gripper openness
        # Store start states for interpolation
        self.start_lee_tf = self.lee_tf   # Start left end effector transform
        self.start_ree_tf = self.ree_tf   # Start right end effector transform
        self.start_left_gripper = 1.0      # Start left gripper openness
        self.start_right_gripper = 1.0     # Start right gripper openness

        # INTERACTIVE_NEW mode: trajectory queue for 30-step execution
        self.trajectory_queue_size = 30
        # Pre-allocate queues to avoid memory allocation during runtime
        # Each queue element is a wp.transform (7 floats: 3 pos + 4 quat)
        self.lee_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.ree_tf_queue = [wp.transform() for _ in range(self.trajectory_queue_size)]
        self.queue_executing = False  # Whether we are executing a queued trajectory
        self.queue_index = 0  # Current index in the queue
        # Cache current end effector targets to avoid GPU->CPU sync
        self.cached_lee_target_tf = self.lee_tf  # Current left end effector target
        self.cached_ree_target_tf = self.ree_tf  # Current right end effector target
        # Pre-compute interpolation alphas for trajectory generation (optimization)
        self.trajectory_alphas = np.linspace(1.0/self.trajectory_queue_size, 1.0, self.trajectory_queue_size, dtype=np.float32)

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
            self._filter_contacts_for_open_gripper(self.contacts)
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
            self._filter_contacts_for_open_gripper(self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # Swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives with displacement limiting."""
        # Clamp left end effector displacement
        clamped_lee_tf = clamp_transform_delta(
            self.prev_gizmo_lee_tf, 
            self.gizmo_lee_tf, 
            self.max_displacement_per_frame, 
            self.max_rotation_per_frame
        )
        
        # Clamp right end effector displacement
        clamped_ree_tf = clamp_transform_delta(
            self.prev_gizmo_ree_tf, 
            self.gizmo_ree_tf, 
            self.max_displacement_per_frame, 
            self.max_rotation_per_frame
        )
        
        # Set IK targets with clamped transforms
        self.l_pos_obj.set_target_position(0, wp.transform_get_translation(clamped_lee_tf))
        q = wp.transform_get_rotation(clamped_lee_tf)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        self.r_pos_obj.set_target_position(0, wp.transform_get_translation(clamped_ree_tf))
        q = wp.transform_get_rotation(clamped_ree_tf)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        
        # Update previous transforms for next frame
        self.prev_gizmo_lee_tf = clamped_lee_tf
        self.prev_gizmo_ree_tf = clamped_ree_tf

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                self.open_left_gripper -= 0.1
            else:
                self.open_left_gripper += 0.1

            if self.viewer.is_key_down("2"):
                self.open_right_gripper -= 0.1
            else:
                self.open_right_gripper += 0.1
            self.open_left_gripper = np.clip(self.open_left_gripper, 0.0, 1.0)
            self.open_right_gripper = np.clip(self.open_right_gripper, 0.0, 1.0)
            # Commented out for better performance (avoids CPU-GPU sync)
            # print(f"Open left gripper: {self.open_left_gripper}, Open right gripper: {self.open_right_gripper}")
            # print(f"Left gripper state: {self.left_gripper_state}, Right gripper state: {self.right_gripper_state}")

        # Commented out for better performance
        # print(f"Left  end effector (clamped):{clamped_lee_tf}")
        # print(f"Right end effector (clamped):{clamped_ree_tf}")

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

    def _has_gizmo_moved(self, threshold_pos=0.001, threshold_rot=0.01):
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

    def _push_targets_from_gizmos_new(self):
        """Read gizmo-updated transform and create a 30-step trajectory queue.
        
        This method allows continuous gizmo input and creates smooth interpolated trajectories:
        - Always reads gizmo input from viewer
        - When gizmo moves significantly, creates a new 30-step trajectory for end effectors
        - Gripper openness is applied immediately without queueing
        - Executes one step from the queue per frame
        - Allows user to interrupt and create new trajectories at any time
        """
        # Handle gripper control via keyboard (immediate response, no queue)
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                self.open_left_gripper -= 0.1
            else:
                self.open_left_gripper += 0.1

            if self.viewer.is_key_down("2"):
                self.open_right_gripper -= 0.1
            else:
                self.open_right_gripper += 0.1
                
            self.open_left_gripper = np.clip(self.open_left_gripper, 0.0, 1.0)
            self.open_right_gripper = np.clip(self.open_right_gripper, 0.0, 1.0)
        
        # Check if gizmo has moved significantly
        gizmo_moved = self._has_gizmo_moved()
        
        # Create new trajectory if gizmo moved or if not currently executing
        if gizmo_moved or not self.queue_executing:
            # Clamp left end effector displacement
            clamped_lee_tf = clamp_transform_delta(
                self.prev_gizmo_lee_tf, 
                self.gizmo_lee_tf, 
                self.max_displacement_per_frame * self.trajectory_queue_size,  # Scale for total movement
                self.max_rotation_per_frame * self.trajectory_queue_size
            )
            
            # Clamp right end effector displacement
            clamped_ree_tf = clamp_transform_delta(
                self.prev_gizmo_ree_tf, 
                self.gizmo_ree_tf, 
                self.max_displacement_per_frame * self.trajectory_queue_size,
                self.max_rotation_per_frame * self.trajectory_queue_size
            )
            
            # Use cached target transforms to avoid GPU->CPU sync
            # These represent the last target we set for IK, avoiding expensive .numpy() call
            current_lee_tf = self.cached_lee_target_tf
            current_ree_tf = self.cached_ree_target_tf
            
            # Fill pre-allocated trajectory queues by interpolating from current to target
            # Optimized: inline interpolation to reduce function call overhead
            # Note: For 30 iterations, function call overhead is ~10-20% of total time
            lee_pos_start = wp.transform_get_translation(current_lee_tf)
            lee_pos_end = wp.transform_get_translation(clamped_lee_tf)
            lee_rot_start = wp.transform_get_rotation(current_lee_tf)
            lee_rot_end = wp.transform_get_rotation(clamped_lee_tf)
            
            ree_pos_start = wp.transform_get_translation(current_ree_tf)
            ree_pos_end = wp.transform_get_translation(clamped_ree_tf)
            ree_rot_start = wp.transform_get_rotation(current_ree_tf)
            ree_rot_end = wp.transform_get_rotation(clamped_ree_tf)
            
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
            self.queue_index = 0
            
            # Update previous transforms
            self.prev_gizmo_lee_tf = clamped_lee_tf
            self.prev_gizmo_ree_tf = clamped_ree_tf
            
            # Skip debug print for better GPU performance
            # if gizmo_moved:
            #     print(f"Gizmo moved! Creating new trajectory with {self.trajectory_queue_size} steps")
        
        # Execute current step in the queue (only for end effectors)
        if self.queue_executing and self.queue_index < len(self.lee_tf_queue):
            # Get current target from queue
            target_lee_tf = self.lee_tf_queue[self.queue_index]
            target_ree_tf = self.ree_tf_queue[self.queue_index]
            
            # Set IK targets
            self.l_pos_obj.set_target_position(0, wp.transform_get_translation(target_lee_tf))
            q = wp.transform_get_rotation(target_lee_tf)
            self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
            
            self.r_pos_obj.set_target_position(0, wp.transform_get_translation(target_ree_tf))
            q = wp.transform_get_rotation(target_ree_tf)
            self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
            
            # Update cached targets (avoids GPU sync in next trajectory generation)
            self.cached_lee_target_tf = target_lee_tf
            self.cached_ree_target_tf = target_ree_tf
            
            # Advance queue index
            self.queue_index += 1
            
            # Check if queue is finished
            if self.queue_index >= len(self.lee_tf_queue):
                self.queue_executing = False
                self.queue_index = 0

    def _update_disabled_shapes(self):
        """Update GPU array of disabled shapes based on gripper states.
        
        Optimization: Uses pre-computed arrays and only updates GPU when state changes.
        """
        # Determine current state (which grippers are opening)
        left_opening = float(self.left_gripper_state) < float(self.open_left_gripper)
        right_opening = float(self.right_gripper_state) < float(self.open_right_gripper)
        current_state = (left_opening, right_opening)
        
        # Early exit if state hasn't changed (avoids redundant GPU transfer)
        if current_state == self._prev_disabled_state:
            return
        
        # State changed - update GPU arrays using pre-computed cache
        self._prev_disabled_state = current_state
        disabled_array, count = self._disabled_shapes_cache[current_state]
        
        if count > 0:
            self.disabled_shapes_wp.assign(disabled_array)
        self.disabled_shape_count_wp.assign([count])

    def _filter_contacts_for_open_gripper(self, contacts):
        """Remove contacts involving gripper shapes when the corresponding gripper is opening.
        
        This function filters out collision contacts that involve gripper finger shapes when
        the gripper is in the process of opening (releasing). This prevents unwanted collision
        forces when the gripper releases an object.
        
        Left and right grippers are handled independently:
        - Left gripper opening (left_prev < left_target) only disables left gripper collisions
        - Right gripper opening (right_prev < right_target) only disables right gripper collisions
        - They do not affect each other
        
        Implementation details:
        - Uses GPU kernels for all operations (no host-device synchronization)
        - Compatible with CUDA graph capture for performance optimization
        - Uses prefix sum scan for efficient array compaction
        - Processes both rigid contacts (rigid-rigid collisions) and soft contacts (cloth-body collisions)
        
        Args:
            contacts (Contacts): The contact data structure containing rigid and soft contact arrays.
                This structure will be modified in-place by removing contacts involving disabled shapes.
        
        Algorithm:
            1. Filter rigid contacts:
               a. Mark contacts to keep (keep_mask = 1 if both shapes are not disabled)
               b. Compute prefix sum of keep_mask to determine new indices
               c. Compact arrays by copying only kept contacts to their new positions
               d. Update contact count
            
            2. Filter soft contacts (same process as rigid contacts):
               a. Mark contacts to keep (keep_mask = 1 if shape is not disabled)
               b. Compute prefix sum and compact arrays
               c. Update contact count
        
        Note:
            - All operations are performed on GPU without host synchronization
            - The disabled_shapes_wp array must be updated before calling this function
            - Empty results (no contacts to filter) are handled gracefully by kernels
        """
        # ======================================================================
        # Filter Rigid Contacts (rigid-rigid collisions)
        # ======================================================================
        # Step 1: Mark contacts to keep or remove
        # Each thread processes one contact and checks if either shape in the contact
        # pair is in the disabled_shapes list. If so, mark for removal (keep_mask = 0).
        wp.launch(
            kernel=_filter_rigid_contacts_kernel,
            dim=self.model.rigid_contact_max,  # Launch with max dimension, kernel checks count internally
            inputs=[
                contacts.rigid_contact_count,      # Current number of rigid contacts (read from GPU)
                contacts.rigid_contact_shape0,     # First shape index in each contact pair
                contacts.rigid_contact_shape1,     # Second shape index in each contact pair
                self.disabled_shapes_wp,          # Array of disabled shape IDs
                self.disabled_shape_count_wp,     # Number of disabled shapes
                self.keep_mask_wp,                 # Output: 1 to keep, 0 to remove
            ],
            device=self.model.device,
        )
        
        # Step 2: Compute prefix sum for array compaction
        # The prefix sum gives us the new index for each kept contact in the compacted array.
        # Example: if keep_mask = [1, 0, 1, 1, 0], prefix_sum = [1, 1, 2, 3, 3]
        # This tells us: contact 0 -> new index 0, contact 2 -> new index 1, contact 3 -> new index 2
        wp.utils.array_scan(
            self.keep_mask_wp[:self.model.rigid_contact_max], 
            self.prefix_sum_wp[:self.model.rigid_contact_max], 
            inclusive=True  # Inclusive prefix sum: includes current element
        )
        
        # Step 3: Get the new contact count
        # The last element of prefix_sum tells us how many contacts we're keeping.
        # This is computed on GPU to avoid host synchronization.
        wp.launch(
            kernel=_get_new_count_kernel,
            dim=1,
            inputs=[
                self.prefix_sum_wp,                # Prefix sum array
                contacts.rigid_contact_count,      # Original contact count
                self.new_count_wp,                 # Output: new contact count after filtering
            ],
            device=self.model.device,
        )
        
        # Step 4: Compact the contact arrays
        # Copy only the kept contacts to their new positions at the beginning of the arrays.
        # This is done in-place: we overwrite the original arrays with the compacted data.
        # The kernel only processes contacts that are marked to keep (keep_mask = 1).
        wp.launch(
            kernel=_compact_rigid_contacts_kernel,
            dim=self.model.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,      # Current contact count (for bounds checking)
                self.keep_mask_wp,                 # Mask indicating which contacts to keep
                self.prefix_sum_wp,                # Prefix sum for new index calculation
                # Input arrays (source data)
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,     # Contact point on first shape
                contacts.rigid_contact_point1,     # Contact point on second shape
                contacts.rigid_contact_normal,    # Contact normal vector
                contacts.rigid_contact_thickness0, # Thickness of first shape at contact
                contacts.rigid_contact_thickness1, # Thickness of second shape at contact
                # Output arrays (same as input, overwritten in-place)
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
            ],
            device=self.model.device,
        )
        
        # Step 5: Update the contact count
        # Copy the new count from GPU array to the contacts structure.
        # This is the only GPU-to-GPU copy operation, no host sync required.
        contacts.rigid_contact_count.assign(self.new_count_wp)

        # ======================================================================
        # Filter Soft Contacts (cloth-body collisions)
        # ======================================================================
        # Same process as rigid contacts, but for soft contacts (particle-shape collisions).
        # Soft contacts only have one shape per contact (the rigid body shape),
        # so we only need to check if that shape is disabled.
        
        # Step 1: Mark soft contacts to keep or remove
        # Check if the shape in each contact is disabled.
        wp.launch(
            kernel=_filter_soft_contacts_kernel,
            dim=self.soft_contact_max,
            inputs=[
                contacts.soft_contact_count,      # Current number of soft contacts
                contacts.soft_contact_shape,       # Shape index for each soft contact
                self.disabled_shapes_wp,          # Array of disabled shape IDs
                self.disabled_shape_count_wp,     # Number of disabled shapes
                self.keep_mask_wp,                 # Output: 1 to keep, 0 to remove
            ],
            device=self.model.device,
        )
        
        # Step 2: Compute prefix sum for compaction
        wp.utils.array_scan(
            self.keep_mask_wp[:self.soft_contact_max], 
            self.prefix_sum_wp[:self.soft_contact_max], 
            inclusive=True
        )
        
        # Step 3: Get the new contact count
        wp.launch(
            kernel=_get_new_count_kernel,
            dim=1,
            inputs=[
                self.prefix_sum_wp,
                contacts.soft_contact_count,
                self.new_count_wp,
            ],
            device=self.model.device,
        )
        
        # Step 4: Compact the soft contact arrays
        # Copy only kept contacts to their new positions.
        wp.launch(
            kernel=_compact_soft_contacts_kernel,
            dim=self.soft_contact_max,
            inputs=[
                contacts.soft_contact_count,
                self.keep_mask_wp,
                self.prefix_sum_wp,
                # Input arrays (source data)
                contacts.soft_contact_particle,   # Particle index in the contact
                contacts.soft_contact_shape,       # Shape index in the contact
                contacts.soft_contact_body_pos,    # Body position at contact
                contacts.soft_contact_body_vel,    # Body velocity at contact
                contacts.soft_contact_normal,      # Contact normal vector
                # Output arrays (same as input, overwritten in-place)
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
            ],
            device=self.model.device,
        )
        
        # Step 5: Update the soft contact count
        contacts.soft_contact_count.assign(self.new_count_wp)

    def _push_targets_from_trajectories(self):
        """Read transform from trajectory and push into IK objectives.
        
        Optimization: Direct unpacking and minimal intermediate operations.
        """
        # Left gripper trajectory
        transform_data, state = self.trajectory_animation.get_pose("left_gripper", self.sim_time)
        lee_transform = wp.transform(*transform_data)
        lee_pos = wp.transform_get_translation(lee_transform)
        lee_rot = wp.transform_get_rotation(lee_transform)
        self.l_pos_obj.set_target_position(0, lee_pos)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(lee_rot[0], lee_rot[1], lee_rot[2], lee_rot[3]))
        self.open_left_gripper = state

        # Right gripper trajectory
        transform_data, state = self.trajectory_animation.get_pose("right_gripper", self.sim_time)
        ree_transform = wp.transform(*transform_data)
        ree_pos = wp.transform_get_translation(ree_transform)
        ree_rot = wp.transform_get_rotation(ree_transform)
        self.r_pos_obj.set_target_position(0, ree_pos)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(ree_rot[0], ree_rot[1], ree_rot[2], ree_rot[3]))
        self.open_right_gripper = state

    def step(self):
        """Execute one simulation step."""
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.left_gripper_state = self.open_left_gripper
            self.right_gripper_state = self.open_right_gripper

        # Avoid print in INTERACTIVE_NEW mode for better GPU performance
        if self.animation_type != AnimationType.INTERACTIVE_NEW:
            print('Step time:', self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            self._push_targets_from_gizmos()

        if self.animation_type == AnimationType.INTERACTIVE_NEW:
            self._push_targets_from_gizmos_new()

        if self.animation_type == AnimationType.TRAJECTORY:
            self._push_targets_from_trajectories()

        # Update disabled shapes array before physics simulation (for CUDA graph compatibility)
        self._update_disabled_shapes()

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
            # Print FPS info (skip frequent printing in INTERACTIVE_NEW mode for better performance)
            if self.animation_type != AnimationType.INTERACTIVE_NEW or self.sim_frame % 30 == 0:
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
        
        if self.animation_type == AnimationType.INTERACTIVE_NEW:
            # Register gizmo for INTERACTIVE_NEW mode
            # Always accept gizmo input, will create new trajectory when moved
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
        
        self.viewer.log_state(self.state_0)

        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        # Only synchronize for non-GPU-optimized modes
        # For INTERACTIVE_NEW, rely on implicit synchronization at render time
        if self.animation_type != AnimationType.INTERACTIVE_NEW:
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

    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
