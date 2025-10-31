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

# ---------------------------------------------
# Warp kernels for substep joint interpolation
# ---------------------------------------------

@wp.kernel
def interpolate_joint_q_kernel(
    start_q: wp.array(dtype=float),
    target_q: wp.array(dtype=float),
    substep_q: wp.array(dtype=float),
    joint_count: int,
    substeps: int,
):
    tid = wp.tid()
    total = joint_count * substeps
    if tid >= total:
        return

    s = tid // joint_count
    j = tid - s * joint_count

    f = float(s + 1) / float(substeps)
    val = start_q[j] + f * (target_q[j] - start_q[j])
    substep_q[tid] = val


@wp.kernel
def write_substep_control_kernel(
    substep_q: wp.array(dtype=float),            # length: substeps * joint_coord_count
    frame_target_q: wp.array(dtype=float),       # length: joint_coord_count
    control_joint_target: wp.array(dtype=float), # length: joint_dof_count
    gripper_vel: wp.array(dtype=float),          # length: joint_dof_count
    pos_mask: wp.array(dtype=int),               # DOF-level mask: position-controlled DOFs
    vel_mask: wp.array(dtype=int),               # DOF-level mask: velocity-controlled DOFs
    interp_pos_mask_dof: wp.array(dtype=int),    # DOF-level mask: DOFs to interpolate (grippers only)
    joint_dof_count: int,
    joint_coord_count: int,
    substeps: int,
    s: int,
):
    j = wp.tid()
    if j >= joint_dof_count:
        return

    if vel_mask[j] != 0:
        control_joint_target[j] = gripper_vel[j]
        return

    if pos_mask[j] != 0:
        if interp_pos_mask_dof[j] != 0:
            control_joint_target[j] = substep_q[s * joint_coord_count + j]
        else:
            control_joint_target[j] = frame_target_q[j]
        return

    # leave unchanged for joints without control

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
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

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
        self.debug_substeps = True

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

        # ---------------------------------------------
        # Buffers and masks for substep control
        # ---------------------------------------------
        self.joint_coord_count = int(self.model.joint_coord_count)
        self.joint_dof_count = int(self.model.joint_dof_count)

        # substep joint positions (flattened: substeps * joint_coord_count)
        self.substep_joint_q = wp.zeros(self.sim_substeps * self.joint_coord_count, dtype=float)
        # per-frame target joint positions (length = joint_coord_count)
        self.frame_target_q = wp.zeros(self.joint_coord_count, dtype=float)

        # control masks
        pos_mask_np = np.zeros(self.joint_dof_count, dtype=np.int32)
        vel_mask_np = np.zeros(self.joint_dof_count, dtype=np.int32)
        # controllable arm joints always in position-target mode
        pos_mask_np[self.controllable_joint_indices] = 1
        # gripper control depends on mode
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            pos_mask_np[self.left_gripper_joint_indices] = 1
            pos_mask_np[self.right_gripper_joint_indices] = 1
        elif self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
            vel_mask_np[self.left_gripper_joint_indices] = 1
            vel_mask_np[self.right_gripper_joint_indices] = 1

        self.pos_mask = wp.array(pos_mask_np, dtype=int)
        self.vel_mask = wp.array(vel_mask_np, dtype=int)
        # per-frame gripper velocity targets (length = joint_dof_count)
        self.gripper_vel = wp.zeros(self.joint_dof_count, dtype=float)

        # which DOFs to interpolate across substeps (grippers only)
        interp_pos_mask_np = np.zeros(self.joint_dof_count, dtype=np.int32)
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            interp_pos_mask_np[self.left_gripper_joint_indices] = 1
            interp_pos_mask_np[self.right_gripper_joint_indices] = 1
        self.interp_pos_mask_dof = wp.array(interp_pos_mask_np, dtype=int)

        # ------------------------------------------------------------------
        # End effector
        # ------------------------------------------------------------------
        self.open_left_gripper = 1.0
        self.open_right_gripper = 1.0

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
        # Skip CUDA graph capture when debugging substeps to avoid host copies during capture
        if wp.get_device().is_cuda and not self.use_mujoco_cpu and not self.debug_substeps:
            with wp.ScopedCapture() as capture:
                self.physics_simulate()
            self.physics_graph = capture.graph

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        for s in range(self.sim_substeps):
            # assign per-substep joint targets to control
            wp.launch(
                write_substep_control_kernel,
                dim=self.joint_dof_count,
                inputs=[
                    self.substep_joint_q,
                    self.frame_target_q,
                    self.control.joint_target,
                    self.gripper_vel,
                    self.pos_mask,
                    self.vel_mask,
                    self.interp_pos_mask_dof,
                    self.joint_dof_count,
                    self.joint_coord_count,
                    self.sim_substeps,
                    s,
                ],
            )
            if self.debug_substeps:
                # Debug: print control targets and expected substep positions for grippers
                ct_np = self.control.joint_target.numpy()
                ft_np = self.frame_target_q.numpy()
                ss_np = self.substep_joint_q.numpy()
                q_before_np = self.state_0.joint_q.numpy()

                ss_slice = ss_np[s*self.joint_coord_count:(s+1)*self.joint_coord_count]

                left_ct = ct_np[self.left_gripper_joint_indices]
                right_ct = ct_np[self.right_gripper_joint_indices]
                left_ft = ft_np[self.left_gripper_joint_indices]
                right_ft = ft_np[self.right_gripper_joint_indices]
                left_ss = ss_slice[self.left_gripper_joint_indices]
                right_ss = ss_slice[self.right_gripper_joint_indices]
                left_q0 = q_before_np[self.left_gripper_joint_indices]
                right_q0 = q_before_np[self.right_gripper_joint_indices]

                print(f"[substep {s}] targets L:{left_ct} R:{right_ct}  frame L:{left_ft} R:{right_ft}  substep L:{left_ss} R:{right_ss}  q_before L:{left_q0} R:{right_q0}")
            self.contacts = self.model.collide(self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # control assigned above per substep

            # Clear particle info for rigid_solver
            particle_count = self.model.particle_count
            self.model.particle_count = 0

            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)


            # Recover the particle info
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            # Solve the cloth, add force onto state_1
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
            if self.debug_substeps:
                q_after_np = self.state_0.joint_q.numpy()
                left_q1 = q_after_np[self.left_gripper_joint_indices]
                right_q1 = q_after_np[self.right_gripper_joint_indices]
                print(f"[substep {s}] q_after  L:{left_q1} R:{right_q1}")

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
                self.open_left_gripper = 0.0
            else:
                self.open_left_gripper = 1.0

            if self.viewer.is_key_down("2"):
                self.open_right_gripper = 0.0
            else:
                self.open_right_gripper = 1.0

        print(f"Left  end effector:{self.lee_tf}")
        print(f"Right end effector:{self.ree_tf}")

        # self.cloth_solver.finger_indices.assign([self.open_left_gripper, self.open_left_gripper, self.open_right_gripper, self.open_right_gripper])

        # self.cloth_solver.finger_states.assign([self.open_left_gripper, self.open_left_gripper, self.open_right_gripper, self.open_right_gripper])

        # print(self.cloth_solver.finger_states)

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
        print('step time', self.sim_time)

        if self.animation_type == AnimationType.INTERACTIVE:
            self._push_targets_from_gizmos()

        if self.animation_type == AnimationType.TRAJECTORY:
            self._push_targets_from_trajectories()


        # IK step, update self.ik_joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        ik_joint_q = self.ik_joint_q.flatten()
        ik_joint_q_np = ik_joint_q.numpy()


        # Position control for [gripper]
        joint_limit_lower_np = self.model.joint_limit_lower.numpy()
        joint_limit_upper_np = self.model.joint_limit_upper.numpy()
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            # openness in [0,1] maps linearly to [lower, upper]
            left_lower = joint_limit_lower_np[self.left_gripper_joint_indices]
            left_upper = joint_limit_upper_np[self.left_gripper_joint_indices]
            right_lower = joint_limit_lower_np[self.right_gripper_joint_indices]
            right_upper = joint_limit_upper_np[self.right_gripper_joint_indices]

            left_pos = left_lower + self.open_left_gripper * (left_upper - left_lower)
            right_pos = right_lower + self.open_right_gripper * (right_upper - right_lower)

            ik_joint_q_np[self.left_gripper_joint_indices] = left_pos
            ik_joint_q_np[self.right_gripper_joint_indices] = right_pos
        ik_joint_q.assign(ik_joint_q_np)

        # Align the self.state with the target body in the viewer
        # newton.eval_fk(self.model, ik_joint_q, self.model.joint_qd, self.state)
        
        # Limit the joint movement in one frame
        move, target = limit_joint_move(ik_joint_q.numpy(), self.state_0.joint_q.numpy(), 20.0, self.frame_dt)

        # Prepare per-frame targets for substep interpolation (GPU)
        self.frame_target_q.assign(target.flatten())

        # Set joint qd for [controllable joint] and [gripper]
        joint_qd_np = self.state_0.joint_qd.numpy()
        ik_joint_qd_np = move/self.frame_dt
        joint_qd_np[self.controllable_joint_indices] = ik_joint_qd_np[self.controllable_joint_indices]
        self.ik_joint_qd.assign(joint_qd_np)
        self.state_0.joint_qd.assign(joint_qd_np)

        # Set joint velocity for the grippers
        gripper_vel = 0.2

        # Velocity control for [gripper] (per-frame, replicated across substeps on GPU)
        if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
            gv_np = np.zeros(self.joint_dof_count, dtype=np.float32)
            vel = linear_map(self.open_left_gripper, -gripper_vel, gripper_vel)
            gv_np[self.left_gripper_joint_indices] = vel
            vel = linear_map(self.open_right_gripper, -gripper_vel, gripper_vel)
            gv_np[self.right_gripper_joint_indices] = vel
            self.gripper_vel.assign(gv_np)

        # Precompute substep joint targets on GPU
        wp.launch(
            interpolate_joint_q_kernel,
            dim=self.sim_substeps * self.joint_coord_count,
            inputs=[
                self.state_0.joint_q,
                self.frame_target_q,
                self.substep_joint_q,
                self.joint_coord_count,
                self.sim_substeps,
            ],
        )

        # Physics step
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
