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

import newton
import newton.examples
import newton.ik as ik
import newton.utils

def transform_diff(tf1, tf2, pos_thres=0.01, rot_thres=0.01):
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


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base) + ground
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        # franka.default_shape_cfg.density = 100.0
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        franka.add_ground_plane()

        self.robot_joint_q_cnt = len(franka.joint_q)

        # Configure target position control for arm joints.
        for i in range(self.robot_joint_q_cnt):
            franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            # franka.joint_target_ke[i] = 600.0
            # franka.joint_target_kd[i] = 10.0
            # franka.joint_target_ke[i] = 1000.0
            franka.joint_target_ke[i] = 1500.0
            franka.joint_target_kd[i] = 5.0

        # Disable joint control for finger joints.
        for i in range(self.robot_joint_q_cnt-2, self.robot_joint_q_cnt):
            franka.joint_dof_mode[i] = newton.JointMode.NONE
            franka.joint_target_ke[i] = 100.0
            franka.joint_target_kd[i] = 10.0
        
        # Set initial pose
        franka.joint_q[:7] = [
            0, -0.25, 0, -1.5,
            0, 1.2, -0.8
        ]

        # Add a box
        pos = wp.vec3(0.5, 0.0, 0.13)
        rot = wp.quat_identity()
        body_box = franka.add_body(xform=wp.transform(p=pos, q=rot))
        franka.add_joint_free(body_box)
        franka.add_shape_box(body_box, hx=0.03, hy=0.03, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=1.0))

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
        self.ee_index = 10  # hardcoded for now
        self.lf_index = 7 # left finger
        self.rf_index = 8 # left finger
        self.open_gripper = 1

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])
        self.old_ee_tf = wp.transform(*body_q_np[self.ee_index])

        # ------------------------------------------------------------------
        # IK setup (single problem, single EE)
        # ------------------------------------------------------------------
        # residual layout:
        # [0..2]  : position (3)
        # [3..5]  : rotation (3)
        # [6..]   : joint limits (joint_coord_count)
        total_residuals = 6 + self.model.joint_coord_count

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position objective
        self.pos_obj = ik.IKPositionObjective(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=0,
        )

        # Rotation objective
        self.rot_obj = ik.IKRotationObjective(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=3,
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=6,
            weight=10.0,
        )

        # Variables the solver will update
        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        print("self.joint_q", self.joint_q)
        print("self.body_mass", self.model.body_mass)

        self.ik_iters = 24
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.joint_q,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            # nefc_per_env=500,
            # njmax=500,
            # ncon_per_env=500,
            solver='newton',
            # impratio=100,
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
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)


    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        self.pos_obj.set_target_position(0, wp.transform_get_translation(self.ee_tf))
        q = wp.transform_get_rotation(self.ee_tf)
        q_normalized = wp.normalize(q)
        
        self.rot_obj.set_target_rotation(0, wp.vec4(q_normalized[0], q_normalized[1], q_normalized[2], q_normalized[3]))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self.sim_time += self.frame_dt
        self._push_targets_from_gizmos()

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("e"):
                self.open_gripper = 0
            else:
                self.open_gripper = 1

        # IK step, update self.joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        # Update control targets according to self.joint_q
        joint_q_np = self.joint_q.numpy()
        
        # Control the arm joints using control.joint_target
        self.joint_q[0,0:self.lf_index].assign(joint_q_np[0,0:self.lf_index])

        newton.eval_fk(self.model, self.joint_q.flatten(), self.model.joint_qd, self.state)
        self.control.joint_target[0:self.lf_index].assign(self.joint_q.flatten()[0:self.lf_index])

        # Control the finger joints using control.joint_f
        # Apply a force of -0.05 to grasp, 1 to release
        self.control.joint_f[self.lf_index:self.lf_index+1].assign([
            linear_map(self.open_gripper, -0.05, 1)
        ])
        self.control.joint_f[self.lf_index:self.rf_index+1].assign([
            linear_map(self.open_gripper, -0.05, 1),
        ])
        # joint_q_np[0, self.lf_index] = (self.open_gripper) * 0.1
        # joint_q_np[0, self.rf_index] = (self.open_gripper) * 0.1

        # Physics step
        if self.physics_graph:
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()

        self.sim_time += self.frame_dt

        ## Optionally reset ee_tf if it deviates too much from the IK result.
        ## This helps keep the end-effector transform consistent, but may not always work as intended.
        # if transform_diff(self.ee_tf, wp.transform(*self.state.body_q.numpy()[self.ee_index])):
        #     self.ee_tf = self.old_ee_tf
        # else:
        #     self.old_ee_tf = self.ee_tf

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmo (viewer will draw & mutate transform in-place)
        self.viewer.log_gizmo("target_tcp", self.ee_tf)
        
        # Visualize the current articulated state
        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

        wp.synchronize()





if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
