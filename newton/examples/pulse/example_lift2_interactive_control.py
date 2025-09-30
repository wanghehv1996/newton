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

import newton
import newton.examples
import newton.ik as ik
import newton.utils

def limit_joint_move(tar_q, cur_q, max_qd, dt):
    err = tar_q-cur_q
    max_qd = max_qd*dt
    min_qd = -1 * max_qd
    
    delta_q = np.clip(err*0.8,min_qd,max_qd)

    return delta_q, delta_q + cur_q

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

class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.gripper_control_type = GripperControlType.TARGET_POSITION

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base) + ground
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        
        franka.add_urdf(
            # lift2 urdf can be downloaded from https://gitee.pjlab.org.cn/L2/wanghui1/PulseAsset.git
            newton.examples.get_asset("lift2_urdf/robot.urdf"),
            floating=False,
            enable_self_collisions=False,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.17))
        )
        franka.add_ground_plane()

        self.robot_joint_q_cnt = len(franka.joint_q)

        # Configure target position control for arm joints.
        for i in range(self.robot_joint_q_cnt):
            franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            franka.joint_target_ke[i] = 3000.0
            franka.joint_target_kd[i] = 10.0

        # Remove control for the base
        for i in (0,1,2):
            franka.joint_dof_mode[i] = newton.JointMode.NONE
            franka.joint_limit_lower[i] = 0
            franka.joint_limit_upper[i] = 0

        # Remove control for the fingers
        # TODO: control the fingers
        for i in (15,16,23,24):
            # Leave a small gap to avoid penetration
            franka.joint_limit_lower[i] = 0.001
            franka.joint_limit_upper[i] = 0.04

            # Configure target control for finger joints
            if self.gripper_control_type == GripperControlType.NONE:
                franka.joint_dof_mode[i] = newton.JointMode.NONE

            if self.gripper_control_type == GripperControlType.TARGET_POSITION:
                franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
                franka.joint_target_ke[i] = 3000.0
                franka.joint_target_kd[i] = 10.0

            if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
                franka.joint_dof_mode[i] = newton.JointMode.TARGET_VELOCITY
                franka.joint_target_kd[i] = 10.0

        print(f"joint dof={franka.joint_dof_count}, coord = {franka.joint_coord_count}")
        for i in range(franka.body_count):
            print(f"body {i}, key={franka.body_key[i]}")

        for i in range(franka.joint_count):
            print(f"joint {i}, key={franka.joint_key[i]}, type={franka.joint_type[i]}, {franka.joint_dof_dim[i]} {franka.joint_parent[i]} -> {franka.joint_child[i]}")
        
        # Set indices for the end-effector and fingers
        self.lee_index = 12  # left hand end effector body index
        self.lee_lf_index = 15 # left finger joint index
        self.lee_rf_index = 16 # right finger joint index

        self.ree_index = 21  # right hand end effector body index
        self.ree_lf_index = 23 # left finger joint index
        self.ree_rf_index = 24 # right finger joint index

        self.controllable_joints_cnt = self.robot_joint_q_cnt

        # Add a table (tofixed)
        # pos = wp.vec3(0.6, 0.0, 0.0)
        # rot = wp.quat_identity()
        # body_box = franka.add_body()
        # franka.add_joint_fixed(parent=-1, child=body_box, parent_xform=wp.transform(p=pos, q=rot))
        # franka.add_shape_box(body_box, hx=0.4, hy=0.4, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))

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

        # Warp compute graphs
        self.ik_graph = None
        self.physics_graph = None

        # Finialize builder
        self.model = franka.finalize()
        self.model.ground = True

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True

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
        # IK setup (single problem, single EE)
        # ------------------------------------------------------------------
        # residual layout:
        # [0..2]  : position (3)
        # [3..5]  : rotation (3)
        # [6..]   : joint limits (joint_coord_count)
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
        self.ik_iters = 24

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
            njmax=500,
            ncon_per_env=500,
            solver='newton',
            cone="elliptic",
            contact_stiffness_time_const=self.sim_dt # important param to ensure zero penetration
        )

        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.ik_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_simulate()
            self.ik_graph = capture.graph

        self.physics_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.physics_simulate()
            self.physics_graph = capture.graph

    def ik_simulate(self):
        self.solver.solve(iterations=self.ik_iters)

    def physics_simulate(self):
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()
            # TODO:add control
            # self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.rigid_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

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

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self.sim_time += self.frame_dt
        self._push_targets_from_gizmos()

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("1"):
                self.open_left_gripper = 0
            else:
                self.open_left_gripper = 1

            if self.viewer.is_key_down("2"):
                self.open_right_gripper = 0
            else:
                self.open_right_gripper = 1

        # IK step, update self.ik_joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()


        ik_joint_q = self.ik_joint_q.flatten()

        # print(ik_joint_q.shape)

        # Align the self.state with the target body in the viewer
        newton.eval_fk(self.model, ik_joint_q, self.model.joint_qd, self.state)

        # Reset base, finger 
        self.state_0.joint_q[0:3].assign(ik_joint_q[0:3])
        self.state_0.joint_q[15:17].assign(ik_joint_q[15:17])
        self.state_0.joint_q[23:25].assign(ik_joint_q[23:25])

        
        # Limit the joint movement for the arm in one frame
        move, target = limit_joint_move(ik_joint_q[0:self.controllable_joints_cnt].numpy(), self.state_0.joint_q[0:self.controllable_joints_cnt].numpy(), 20.0, self.frame_dt)

        # print("ik q",self.ik_joint_q)
        # print("target",target)
        # print("move", move)

        # Set joint target position for the arm
        self.control.joint_target[3:15].assign(target.flatten()[3:15])
        self.control.joint_target[17:23].assign(target.flatten()[17:23])

        # Manually reset base joint
        self.state_0.joint_qd[0:3].assign(ik_joint_q[0:3]*0.0)

        # Manually reset joint velocity to stablize the simulation (Not sure why)
        self.state_0.joint_qd[3:15].assign(move[3:15]/self.frame_dt)
        self.state_0.joint_qd[17:23].assign(move[17:23]/self.frame_dt)


        # Set joint velocity for the fingers
        gripper_vel = 0.2

        # Position control
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            
            # LEFT
            # Get IK target
            ik_lf_q = self.model.joint_limit_lower[self.lee_lf_index:self.lee_lf_index+1]
            ik_rf_q = self.model.joint_limit_lower[self.lee_rf_index:self.lee_rf_index+1]
            if self.open_left_gripper:
                ik_lf_q = self.model.joint_limit_upper[self.lee_lf_index:self.lee_lf_index+1]
                ik_rf_q = self.model.joint_limit_upper[self.lee_rf_index:self.lee_rf_index+1]

            ik_joint_q[self.lee_lf_index:self.lee_lf_index+1].assign(ik_lf_q)
            ik_joint_q[self.lee_rf_index:self.lee_rf_index+1].assign(ik_rf_q)

            # Get control signal
            move, target = limit_joint_move(ik_joint_q[self.lee_lf_index:self.lee_rf_index+1].numpy(), self.state_0.joint_q[self.lee_lf_index:self.lee_rf_index+1].numpy(), gripper_vel, self.frame_dt)

            # Set control signal
            self.control.joint_target[self.lee_lf_index:self.lee_rf_index+1].assign(target.flatten())
            self.state_0.joint_qd[self.lee_lf_index:self.lee_rf_index+1].assign(move/self.frame_dt)

            # RIGHT
            # Get IK target
            ik_lf_q = self.model.joint_limit_lower[self.ree_lf_index:self.ree_lf_index+1]
            ik_rf_q = self.model.joint_limit_lower[self.ree_rf_index:self.ree_rf_index+1]
            if self.open_right_gripper:
                ik_lf_q = self.model.joint_limit_upper[self.ree_lf_index:self.ree_lf_index+1]
                ik_rf_q = self.model.joint_limit_upper[self.ree_rf_index:self.ree_rf_index+1]

            ik_joint_q[self.ree_lf_index:self.ree_lf_index+1].assign(ik_lf_q)
            ik_joint_q[self.ree_rf_index:self.ree_rf_index+1].assign(ik_rf_q)

            # Get control signal
            move, target = limit_joint_move(ik_joint_q[self.ree_lf_index:self.ree_rf_index+1].numpy(), self.state_0.joint_q[self.ree_lf_index:self.ree_rf_index+1].numpy(), gripper_vel, self.frame_dt)

            # Set control signal
            self.control.joint_target[self.ree_lf_index:self.ree_rf_index+1].assign(target.flatten())
            self.state_0.joint_qd[self.ree_lf_index:self.ree_rf_index+1].assign(move/self.frame_dt)


        # Velocity control
        if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:

            # LEFT
            self.control.joint_target[self.lee_lf_index:self.lee_lf_index+1].assign([
                linear_map(self.open_left_gripper, -gripper_vel, gripper_vel)
            ])
            self.control.joint_target[self.lee_rf_index:self.lee_rf_index+1].assign([
                linear_map(self.open_left_gripper, -gripper_vel, gripper_vel),
            ])

            # RIGHT
            self.control.joint_target[self.ree_lf_index:self.ree_lf_index+1].assign([
                linear_map(self.open_right_gripper, -gripper_vel, gripper_vel)
            ])
            self.control.joint_target[self.ree_rf_index:self.ree_rf_index+1].assign([
                linear_map(self.open_right_gripper, -gripper_vel, gripper_vel),
            ])

        # Physics step
        if self.physics_graph:
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmo (viewer will draw & mutate transform in-place)
        self.viewer.log_gizmo("left_target_tcp", self.lee_tf)
        self.viewer.log_gizmo("right_target_tcp", self.ree_tf)
        # self.viewer.log_state(self.state)
        self.viewer.log_state(self.state_0)

        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        wp.synchronize()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
