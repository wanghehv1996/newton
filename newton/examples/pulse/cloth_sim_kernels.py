"""
Warp kernel functions for cloth simulation.

This module contains GPU kernel functions used in cloth manipulation simulation:
- Control update kernels for robot joint and gripper control
- Collision detection and contact handling kernels
"""

import warp as wp


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
    """GPU kernel to update robot control targets.
    
    This kernel updates joint targets for:
    - Left gripper joints (interpolated open/close)
    - Right gripper joints (interpolated open/close)
    - Controllable joints (IK-controlled joints)
    
    Args:
        joint_limit_lower: Lower joint limits
        joint_limit_upper: Upper joint limits
        left_indices: Indices of left gripper joints
        right_indices: Indices of right gripper joints
        controllable_indices: Indices of IK-controlled joints
        q0_frame: Joint positions at start of frame
        q_target_frame: Target joint positions for end of frame
        current_q: Current joint positions
        joint_target: Output joint targets
        ik_joint_qd: IK joint velocities
        state_qd: State joint velocities
        left_count: Number of left gripper joints
        right_count: Number of right gripper joints
        controllable_count: Number of controllable joints
        sim_substeps: Number of simulation substeps per frame
        substep_index: Current substep index (0-based)
        sim_dt: Simulation timestep
        gripper_params: [left_prev, left_target, right_prev, right_target] gripper openness values
        gripper_control_type: 1=position control, 2=velocity control
    """
    tid = wp.tid()

    t = float(substep_index + 1) / float(sim_substeps)
    left_prev = gripper_params[0]
    left_target = gripper_params[1]
    right_prev = gripper_params[2]
    right_target = gripper_params[3]
    
    # Linear interpolation for gripper openness
    left_open = (1.0 - t) * left_prev + t * left_target
    right_open = (1.0 - t) * right_prev + t * right_target

    if gripper_control_type == 1:
        # Position control mode
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
        # Velocity control mode
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

    # Update controllable joints (IK-controlled)
    if tid < controllable_count:
        ci = controllable_indices[tid]
        alpha = float(substep_index + 1) / float(sim_substeps)
        target_c = q0_frame[ci] + alpha * (q_target_frame[ci] - q0_frame[ci])
        v_c = (target_c - current_q[ci]) / sim_dt
        joint_target[ci] = target_c
        ik_joint_qd[ci] = v_c
        state_qd[ci] = v_c


@wp.kernel
def _update_gripper_collision_kernel(
    shape_flags: wp.array(dtype=int),
    left_gripper_shapes: wp.array(dtype=int),
    right_gripper_shapes: wp.array(dtype=int),
    left_shape_count: int,
    right_shape_count: int,
    left_opening: int,  # 0=closed/maintain, 1=opening (releasing)
    right_opening: int,  # 0=closed/maintain, 1=opening (releasing)
    collide_particles_flag: int,
):
    """GPU kernel to update gripper collision flags.
    
    When gripper is opening/releasing (state < target), disable collision.
    When gripper is closed/maintaining (state >= target), enable collision.
    
    Args:
        shape_flags: Array of shape collision flags
        left_gripper_shapes: Indices of left gripper shape IDs
        right_gripper_shapes: Indices of right gripper shape IDs
        left_shape_count: Number of left gripper shapes
        right_shape_count: Number of right gripper shapes
        left_opening: 0=closed, 1=opening
        right_opening: 0=closed, 1=opening
        collide_particles_flag: Flag bit for particle collision
    """
    tid = wp.tid()
    
    # Update left gripper shapes
    if tid < left_shape_count:
        shape_id = left_gripper_shapes[tid]
        if left_opening == 1:
            # Opening/releasing: disable collision
            shape_flags[shape_id] = shape_flags[shape_id] & ~collide_particles_flag
        else:
            # Closed/maintaining: enable collision
            shape_flags[shape_id] = shape_flags[shape_id] | collide_particles_flag
    
    # Update right gripper shapes
    if tid < right_shape_count:
        shape_id = right_gripper_shapes[tid]
        if right_opening == 1:
            # Opening/releasing: disable collision
            shape_flags[shape_id] = shape_flags[shape_id] & ~collide_particles_flag
        else:
            # Closed/maintaining: enable collision
            shape_flags[shape_id] = shape_flags[shape_id] | collide_particles_flag

