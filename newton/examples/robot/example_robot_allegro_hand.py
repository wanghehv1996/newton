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
# Example Robot Allegro Hand
#
# Shows how to set up a simulation of a Allegro hand articulation
# from a USD file using newton.ModelBuilder.add_usd().
# We also apply a sinusoidal trajectory to the joint targets and
# apply a continuous rotation to the fixed root joint in the form
# of the joint parent transform. The MuJoCo solver is updated
# about this change in the joint parent transform by calling
# self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES).
#
# Command: python -m newton.examples robot_allegro_hand --num-envs 16
#
###########################################################################

import re

import warp as wp

import newton
import newton.examples
from newton.solvers import SolverNotifyFlags

hand_rotation = wp.normalize(wp.quat(0.283, 0.683, -0.622, 0.258))


@wp.kernel
def move_hand(
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    sim_dt: float,
    # outputs
    joint_target: wp.array(dtype=wp.float32),
    joint_parent_xform: wp.array(dtype=wp.transform),
):
    env_id = wp.tid()
    root_joint_id = env_id * 22
    t = sim_time[env_id]

    root_dof_start = joint_qd_start[root_joint_id]

    # animate the finger joints
    for i in range(20):
        di = root_dof_start + i
        target = wp.sin(t + float(i * 6) * 0.1) * 0.15 + 0.3
        joint_target[di] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    # animate the root joint transform
    q = wp.quat_identity()
    q *= wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    q *= wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -t * 0.02)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)

    # update the sim time
    sim_time[env_id] += sim_dt


class Example:
    def __init__(self, viewer, num_envs=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        self.device = wp.get_device()

        allegro_hand = newton.ModelBuilder()

        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=True,
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
            load_non_physics_prims=True,
        )

        # hide collision shapes for the hand links
        for i, key in enumerate(allegro_hand.shape_key):
            if re.match(".*Robot/.*?/collision", key):
                allegro_hand.shape_flags[i] &= ~newton.ShapeFlags.VISIBLE

        # set joint targets and joint drive gains
        for i in range(len(allegro_hand.joint_dof_mode)):
            allegro_hand.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            allegro_hand.joint_target_ke[i] = 150
            allegro_hand.joint_target_kd[i] = 5
            allegro_hand.joint_target[i] = 0.0

        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.num_envs, spacing=(1, 1, 0))

        builder.add_ground_plane()

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.env_time = wp.zeros(self.num_envs, dtype=wp.float32)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="euler",
            njmax=200,
            ncon_per_env=150,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_cpu=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                move_hand,
                dim=self.num_envs,
                inputs=[
                    self.model.joint_qd_start,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.env_time,
                    self.sim_dt,
                ],
                outputs=[self.control.joint_target, self.model.joint_X_p],
            )

            # # update the solver since we have updated the joint parent transforms
            self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=100, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
