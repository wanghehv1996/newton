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

import warp as wp
import numpy as np
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.utils

import io_util
from trajectory_animation import KeyFrameTrajectoryAnimation
import os

@wp.kernel
def _set_gripper_positions_kernel(
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    left_indices: wp.array(dtype=int),
    right_indices: wp.array(dtype=int),
    left_open: float,
    right_open: float,
    ik_joint_q_flat: wp.array(dtype=float),
    left_count: int,
    right_count: int,
):
    tid = wp.tid()

    if tid < left_count:
        idx = left_indices[tid]
        lo = joint_limit_lower[idx]
        hi = joint_limit_upper[idx]
        ik_joint_q_flat[idx] = lo + left_open * (hi - lo)

    if tid < right_count:
        idx_r = right_indices[tid]
        lo_r = joint_limit_lower[idx_r]
        hi_r = joint_limit_upper[idx_r]
        ik_joint_q_flat[idx_r] = lo_r + right_open * (hi_r - lo_r)

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
    # - 0 -> 1: switch immediately to 1 for the whole frame (no interpolation)
    # - 1 -> 0: interpolate linearly over substeps
    # - equal: keep constant
    if left_prev < left_target:
        wp.printf("substep %d left_prev %.6f left_target %.6f\n", substep_index, left_prev, left_target)
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
            # In velocity mode, the control target is velocity
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


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._substep_index = 0

        self.gripper_control_type = GripperControlType.TARGET_POSITION
        # self.gripper_control_type = GripperControlType.TARGET_VELOCITY

        # TODO: 

        self.use_mujoco_cpu = False
        # self.use_mujoco_cpu = True  # Use MuJoCo-CPU (stable cube grasp)
        # self.use_mujoco_cpu = False # Use MuJoCo-Warp (friction still inaccurate)

        self.animation_type = AnimationType.TRAJECTORY
        # self.animation_type = AnimationType.INTERACTIVE

        # dump visualization image sequence
        self.use_dump_image = False
        # self.use_dump_image = True

        # dump joint q into .npz
        self.use_dump_joint = False

        # VBD parameters
        if self.animation_type == AnimationType.INTERACTIVE:
            self.sim_vbd_iterations = 3    
        if self.animation_type == AnimationType.TRAJECTORY:
            self.sim_vbd_iterations = 3
        # self.sim_vbd_iterations = 3
        #       body-cloth contact
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        #       self-contact
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        self.soft_contact_ke = 500
        self.soft_contact_kd = 5e-3

        self.robot_friction = 1.5
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
        # Maximum displacement per instruction to limit speed
        # This ensures smooth motion even when gizmo moves quickly
        self.max_displacement_per_instruction = 0.05  # meters (5cm per 0.5s = 0.1 m/s max speed)
        self.max_rotation_per_instruction = 0.5  # radians (~28 degrees per 0.5s)
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
            njmax=150000, # large enough to avoid nefc overflow
            ncon_per_world=150000, # large enough to avoid illegal mem access
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
        # self.cloth_solver = newton.solvers.SolverVBD(
        #     self.model,
        #     iterations=self.sim_vbd_iterations,
        #     self_contact_radius=self.self_contact_radius,
        #     self_contact_margin=self.self_contact_margin,
        #     handle_self_contact=True,
        #     vertex_collision_buffer_pre_alloc=32,
        #     edge_collision_buffer_pre_alloc=64,
        #     integrate_with_external_rigid_solver=True,
        #     collision_detection_interval=-1,
        # )
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
        # physics_simulate can now be captured into CUDA graph since filtering logic
        # uses GPU kernels without host sync (.numpy/.assign).

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        for s in range(self.sim_substeps):
            self._substep_index = s

            # update controls for this substep on device
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

            # collide and clear (rigid + soft); filter gripper contacts if opened
            self.contacts = self.model.collide(self.state_0)
            self._filter_contacts_for_open_gripper(self.contacts)
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Clear particle info for rigid_solver
            particle_count = self.model.particle_count
            self.model.particle_count = 0

            # rigid step
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Recover the particle info
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            # cloth step
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
            self._filter_contacts_for_open_gripper(self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def _clamp_target_transform(self, start_tf, desired_tf, max_translation, max_rotation):
        """Clamp target transform to limit displacement per instruction.
        
        Args:
            start_tf: Starting transform
            desired_tf: Desired target transform
            max_translation: Maximum translation distance (meters)
            max_rotation: Maximum rotation angle (radians)
        
        Returns:
            Clamped target transform
        """
        start_pos = wp.transform_get_translation(start_tf)
        desired_pos = wp.transform_get_translation(desired_tf)
        displacement = desired_pos - start_pos
        trans_dist = wp.length(displacement)
        
        # Clamp translation
        if trans_dist > max_translation:
            clamped_pos = start_pos + displacement * (max_translation / trans_dist)
        else:
            clamped_pos = desired_pos
        
        # Clamp rotation
        start_rot = wp.transform_get_rotation(start_tf)
        desired_rot = wp.transform_get_rotation(desired_tf)
        dot = wp.dot(start_rot, desired_rot)
        dot = wp.clamp(dot, -1.0, 1.0)
        rot_angle = 2.0 * wp.acos(abs(dot))
        
        if rot_angle > max_rotation:
            # Interpolate rotation to limit angle
            t = max_rotation / rot_angle
            clamped_rot = wp.quat_slerp(start_rot, desired_rot, t)
        else:
            clamped_rot = desired_rot
        
        return wp.transform(clamped_pos, clamped_rot)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives.
        
        In INTERACTIVE mode, this function:
        1. Detects changes in gizmo positions or gripper states
        2. Clamps displacement to limit maximum speed
        3. Starts a new instruction if changes detected
        4. Each instruction executes smoothly over 0.5 seconds (30 frames at 60 FPS)
        5. Interpolates between start and target states
        6. Automatically disables collision when gripper is opening
        
        Action decomposition:
        - Large movements are automatically decomposed into smaller steps
        - Each step moves at most max_displacement_per_instruction distance
        - Steps execute sequentially, each taking 0.5 seconds
        - This ensures smooth motion regardless of gizmo movement speed
        """
        # Read current gizmo states (gizmos are mutated in-place by viewer in render())
        # Note: gizmo values are updated AFTER step() in render(), so we read the values
        # that were set in the previous render() call
        current_lee_tf = self.gizmo_lee_tf
        current_ree_tf = self.gizmo_ree_tf
        
        # Read keyboard input for gripper control (same logic as init.py)
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                current_left_gripper = 0.0
            else:
                current_left_gripper = 1.0
            
            if self.viewer.is_key_down("2"):
                current_right_gripper = 0.0
            else:
                current_right_gripper = 1.0
        else:
            current_left_gripper = 1.0
            current_right_gripper = 1.0
        
        # Check if any target has changed (gizmo moved or gripper state changed)
        # We compare with current interpolated position (self.lee_tf) instead of target
        # to ensure we always start from the actual current position
        # Use lower threshold to detect even small gizmo movements
        lee_changed = transform_diff(current_lee_tf, self.lee_tf, pos_thres=1e-4, rot_thres=1e-4)
        ree_changed = transform_diff(current_ree_tf, self.ree_tf, pos_thres=1e-4, rot_thres=1e-4)
        left_gripper_changed = (abs(current_left_gripper - self.target_left_gripper) > 1e-6)
        right_gripper_changed = (abs(current_right_gripper - self.target_right_gripper) > 1e-6)
        
        # If any target changed, start a new instruction
        # Always clamp from current position to gizmo position to enforce displacement limit
        if lee_changed or ree_changed or left_gripper_changed or right_gripper_changed:
            # Store current interpolated state as start state
            # This ensures smooth continuation from current position
            self.start_lee_tf = self.lee_tf
            self.start_ree_tf = self.ree_tf
            self.start_left_gripper = self.open_left_gripper
            self.start_right_gripper = self.open_right_gripper
            
            # Clamp targets to limit displacement per instruction
            # Always clamp from current position (self.lee_tf) to gizmo position (current_lee_tf)
            # This automatically decomposes large movements into smaller steps
            clamped_lee_tf = self._clamp_target_transform(
                self.start_lee_tf,  # Start from current actual position
                current_lee_tf,     # Target is gizmo position
                self.max_displacement_per_instruction,
                self.max_rotation_per_instruction
            )
            clamped_ree_tf = self._clamp_target_transform(
                self.start_ree_tf,  # Start from current actual position
                current_ree_tf,     # Target is gizmo position
                self.max_displacement_per_instruction,
                self.max_rotation_per_instruction
            )
            
            # Update targets (clamped to limit speed)
            self.target_lee_tf = clamped_lee_tf
            self.target_ree_tf = clamped_ree_tf
            self.target_left_gripper = current_left_gripper
            self.target_right_gripper = current_right_gripper
            
            # Start new instruction
            self.current_instruction_start_time = self.sim_time
        
        # Interpolate to target based on instruction progress
        # This ensures smooth decomposition: motion happens over 0.5 seconds
        instruction_elapsed = self.sim_time - self.current_instruction_start_time
        t = min(instruction_elapsed / self.instruction_duration, 1.0)  # Clamp to [0, 1]
        
        # Interpolate end effector positions (linear interpolation)
        start_pos_l = wp.transform_get_translation(self.start_lee_tf)
        target_pos_l = wp.transform_get_translation(self.target_lee_tf)
        lerped_pos_l = start_pos_l + (target_pos_l - start_pos_l) * t
        
        start_pos_r = wp.transform_get_translation(self.start_ree_tf)
        target_pos_r = wp.transform_get_translation(self.target_ree_tf)
        lerped_pos_r = start_pos_r + (target_pos_r - start_pos_r) * t
        
        # Interpolate rotations (spherical linear interpolation)
        start_rot_l = wp.transform_get_rotation(self.start_lee_tf)
        target_rot_l = wp.transform_get_rotation(self.target_lee_tf)
        lerped_rot_l = wp.quat_slerp(start_rot_l, target_rot_l, t)
        
        start_rot_r = wp.transform_get_rotation(self.start_ree_tf)
        target_rot_r = wp.transform_get_rotation(self.target_ree_tf)
        lerped_rot_r = wp.quat_slerp(start_rot_r, target_rot_r, t)
        
        # Update transforms (create new transform objects)
        self.lee_tf = wp.transform(lerped_pos_l, lerped_rot_l)
        self.ree_tf = wp.transform(lerped_pos_r, lerped_rot_r)
        
        # Interpolate gripper states (linear interpolation)
        self.open_left_gripper = self.start_left_gripper + (self.target_left_gripper - self.start_left_gripper) * t
        self.open_right_gripper = self.start_right_gripper + (self.target_right_gripper - self.start_right_gripper) * t
        
        # Push interpolated targets to IK objectives (same API as init.py)
        self.l_pos_obj.set_target_position(0, wp.transform_get_translation(self.lee_tf))
        q = wp.transform_get_rotation(self.lee_tf)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        
        self.r_pos_obj.set_target_position(0, wp.transform_get_translation(self.ree_tf))
        q = wp.transform_get_rotation(self.ree_tf)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def _update_disabled_shapes(self):
        """Update GPU array of disabled shapes based on gripper states (CUDA graph compatible)."""
        disabled_shapes = []
        # Left gripper: disable if opening (left_prev < left_target)
        if float(self.left_gripper_state) < float(self.open_left_gripper):
            disabled_shapes.extend(list(self.left_gripper_shape_set))
        # Right gripper: disable if opening (right_prev < right_target)
        if float(self.right_gripper_state) < float(self.open_right_gripper):
            disabled_shapes.extend(list(self.right_gripper_shape_set))
        
        # Update GPU array
        if disabled_shapes:
            self.disabled_shapes_wp.assign(np.array(disabled_shapes, dtype=np.int32))
            self.disabled_shape_count_wp.assign([len(disabled_shapes)])
        else:
            self.disabled_shape_count_wp.assign([0])

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
        """Read transform from trajectory and push into IK objectives."""
        
        transform, state = self.trajectory_animation.get_pose("left_gripper", self.sim_time)
        transform = wp.transform(*transform)
        self.l_pos_obj.set_target_position(0, wp.transform_get_translation(transform))
        q = wp.transform_get_rotation(transform)
        self.l_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        self.open_left_gripper = state

        transform, state = self.trajectory_animation.get_pose("right_gripper", self.sim_time)
        transform = wp.transform(*transform)
        self.r_pos_obj.set_target_position(0, wp.transform_get_translation(transform))
        q = wp.transform_get_rotation(transform)
        self.r_rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))
        self.open_right_gripper = state


        # self.cloth_solver.finger_states.assign([self.open_right_gripper, self.open_right_gripper, self.open_left_gripper, self.open_left_gripper])

        # print(self.cloth_solver.finger_states, self.cloth_solver.finger_indices)

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.sim_time == 0.0:
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.left_gripper_state = self.open_left_gripper
            self.right_gripper_state = self.open_right_gripper

        print('step time', self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            self._push_targets_from_gizmos()

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
        # ik_joint_q_np = ik_joint_q.numpy()
        # joint_limit_lower_np = self.model.joint_limit_lower.numpy()
        # joint_limit_upper_np = self.model.joint_limit_upper.numpy()
        # print('joint_limit_lower', joint_limit_lower_np[self.left_gripper_joint_indices])
        # print('joint_limit_upper', joint_limit_upper_np[self.left_gripper_joint_indices])
        # Cache start-of-frame joint positions and full-frame IK target
        q0_frame = self.state_0.joint_q.numpy()
        q_target_frame = ik_joint_q.numpy()
        # Push per-frame arrays to device
        self.q0_frame_wp.assign(q0_frame.reshape((-1,)))
        self.q_target_frame_wp.assign(q_target_frame.reshape((-1,)))
        # Push gripper params [left_prev, left_target, right_prev, right_target]
        self.gripper_params_wp.assign(np.array([
            float(self.left_gripper_state),
            float(self.open_left_gripper),
            float(self.right_gripper_state),
            float(self.open_right_gripper),
        ], dtype=np.float32))

        # Physics step for all substeps (loop is inside physics_simulate for CUDA graph)
        if self.physics_graph:
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()

        self.sim_time += self.frame_dt
        self.sim_frame += 1
        
        if self.use_dump_joint:
            joint_q_np = self.state_0.joint_q.numpy()
            self.joint_q_seq = np.vstack((self.joint_q_seq, joint_q_np[0:self.robot_joint_q_cnt]))
            self.openness_seq = np.vstack((self.openness_seq, np.array([self.open_left_gripper, self.open_right_gripper])))

            if self.sim_frame == 32 * self.fps:
                np.savez('lift2_manipulating_cloth.npz', joint_q=self.joint_q_seq, openness=self.openness_seq)
        
        self.left_gripper_state = self.open_left_gripper
        self.right_gripper_state = self.open_right_gripper


    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            # Register gizmo (viewer will draw & mutate transform in-place)
            # Use gizmo values directly (they represent user input, not interpolated values)
            # The gizmo transforms are mutated in-place by viewer, so we need to pass the
            # current gizmo values (which will be updated by user interaction)
            self.viewer.log_gizmo("left_target_tcp", self.gizmo_lee_tf)
            self.viewer.log_gizmo("right_target_tcp", self.gizmo_ree_tf)
            # After log_gizmo, viewer has mutated gizmo_lee_tf and gizmo_ree_tf
            # These updated values will be read in the next frame's step()
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
