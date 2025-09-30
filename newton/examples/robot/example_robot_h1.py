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
# Example Robot H1
#
# Shows how to set up a simulation of a H1 articulation
# from a USD file using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_h1 --num-envs 16
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, num_envs=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        self.device = wp.get_device()

        h1 = newton.ModelBuilder()
        h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        h1.default_shape_cfg.ke = 5.0e4
        h1.default_shape_cfg.kd = 5.0e2
        h1.default_shape_cfg.kf = 1.0e3
        h1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd" / "h1_minimal.usda")
        h1.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=True,
            hide_collision_shapes=True,
        )
        # approximate meshes for faster collision detection
        h1.approximate_meshes("bounding_box")

        for i in range(len(h1.joint_dof_mode)):
            h1.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            h1.joint_target_ke[i] = 150
            h1.joint_target_kd[i] = 5

        builder = newton.ModelBuilder()
        builder.replicate(h1, self.num_envs, spacing=(3, 3, 0))

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=100, ls_iterations=50, njmax=100)

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
    parser.add_argument("--num-envs", type=int, default=4, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
