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

from warp.sim.utils import load_mesh
from warp.sim.render import SimRendererUsd

import io_util

# config the joint types
fixed_joint_names = {
    "fixed_base", 
    "root_joint", 
    "fl_fixed_joint",
    "fr_fixed_joint",
    "joint4",
}
controllable_joint_names = {
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

path = "lift2_manipulating_cloth.npz"

class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.use_dump_image = False

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Load trajectory file
        # ------------------------------------------------------------------
        data = np.load(path)
        self.joint_q_seq = data['joint_q']
        self.openness_seq = data['openness']

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

        # # ------------------------------------------------------------------
        # # Set joint groups, print debug info
        # # ------------------------------------------------------------------
        # cnt = 0

        # # body information
        # print("=== Body Information ===")
        # for i in range(franka.body_count):
        #     print(f"body {i}, key={franka.body_key[i]}")
        #     # set left end effector
        #     if franka.body_key[i] in left_ee_body_names:
        #         self.lee_index = i
        #         print(f"  >> left end-effector")
        #     # set right end effector
        #     if franka.body_key[i] in right_ee_body_names:
        #         self.ree_index = i
        #         print(f"  >> right end-effector")

        # # joint information
        # print("=== Joint Information ===")
        # print(f"#joint_dof={franka.joint_dof_count}, #joint_coord = {franka.joint_coord_count}")
        # # joint groups
        # self.fixed_joint_indices = np.array([], dtype = int)
        # self.controllable_joint_indices = np.array([], dtype = int)
        # self.left_gripper_joint_indices = np.array([], dtype = int)
        # self.right_gripper_joint_indices = np.array([], dtype = int)

        # for i in range(franka.joint_count):
        #     print(f"joint {i}, key={franka.joint_key[i]}, type={franka.joint_type[i]}, link={franka.joint_parent[i]} -> {franka.joint_child[i]}, dof_dim={franka.joint_dof_dim[i]}, dof_start {cnt}, dof_lim = [{franka.joint_limit_lower[cnt]}, {franka.joint_limit_upper[cnt]}]")

        #     dof_start = cnt
        #     dof_end = cnt + franka.joint_dof_dim[i][0] + franka.joint_dof_dim[i][1]
        #     # set fixed joint group
        #     if franka.joint_key[i] in fixed_joint_names:
        #         for j in range(dof_start, dof_end):
        #             self.fixed_joint_indices = np.append(self.fixed_joint_indices, [j])
        #     # set controllable joint group
        #     if franka.joint_key[i] in controllable_joint_names:
        #         for j in range(dof_start, dof_end):
        #             self.controllable_joint_indices = np.append(self.controllable_joint_indices, [j])
        #     # set left gripper joint group
        #     if franka.joint_key[i] in left_gripper_joint_names:
        #         for j in range(dof_start, dof_end):
        #             self.left_gripper_joint_indices = np.append(self.left_gripper_joint_indices, [j])
        #     # set right gripper joint group
        #     if franka.joint_key[i] in right_gripper_joint_names:
        #         for j in range(dof_start, dof_end):
        #             self.right_gripper_joint_indices = np.append(self.right_gripper_joint_indices, [j])

        #     cnt += franka.joint_dof_dim[i][0] + franka.joint_dof_dim[i][1]

        # print(f"joint dq cnt check: {cnt} == {franka.joint_dof_count}")
        # print("fixed joint", self.fixed_joint_indices)
        # print("controllable joint", self.controllable_joint_indices)
        # print("left joint", self.left_gripper_joint_indices)
        # print("right joint", self.right_gripper_joint_indices)

        # ------------------------------------------------------------------
        # Finalization and initialization of computational components
        # ------------------------------------------------------------------

        # Finalize builder
        self.model = franka.finalize()
        self.model.ground = True

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            pos = type(self.viewer.camera.pos)(3.0, 0, 1.4)
            self.viewer.camera.pos = pos
            self.viewer.camera.pitch = -20

        # States
        self.state = self.model.state() # for IKSolver
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control() # for control

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.sim_frame < self.joint_q_seq.shape[0]:
            cur_joint_q = self.joint_q_seq[self.sim_frame]
        else:
            cur_joint_q = self.joint_q_seq[-1]
        if self.sim_frame == self.joint_q_seq.shape[0]:
            print(f"reach the end of trajectory, frame = {self.sim_frame}")
        newton.eval_fk(self.model, wp.array(cur_joint_q), self.model.joint_qd, self.state)

        self.sim_time += self.frame_dt
        self.sim_frame += 1

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

        wp.synchronize()
        if self.use_dump_image:
            io_util.dump_gl_frame_image(self.viewer.renderer._screen_width,self.viewer.renderer._screen_height,f"img_{self.sim_frame}.png")

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # parser.set_defaults(viewer="usd", output_path="lift2_interactive_control.usd")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
