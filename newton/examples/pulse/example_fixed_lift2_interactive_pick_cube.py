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
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.gripper_control_type = GripperControlType.TARGET_POSITION
        # self.gripper_control_type = GripperControlType.TARGET_VELOCITY

        # TODO: 
        self.use_mujoco_cpu = True
        # self.use_mujoco_cpu = True  # Use MuJoCo-CPU (stable cube grasp)
        # self.use_mujoco_cpu = False # Use MuJoCo-Warp (friction still inaccurate)

        self.use_dump_image = False

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
            franka.joint_limit_lower[i] = 0.001
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

        # MESH (basket)
        [mesh_vertices, mesh_indices] = load_mesh(newton.examples.get_asset("fruits/plate.obj"))
        mesh_basket = newton.Mesh(mesh_vertices, mesh_indices)
        body_mesh_basket = franka.add_body(xform=wp.transform(p=wp.vec3(0.65, 0, 0.55)))
        franka.add_joint_free(body_mesh_basket)
        franka.add_shape_mesh(body_mesh_basket, mesh = mesh_basket, cfg=newton.ModelBuilder.ShapeConfig(density=710.0))

        # franka.approximate_meshes("coacd")

        # Add a fixed table
        pos = wp.vec3(0.7, 0.0, 0.201)
        rot = wp.quat_identity()
        body_table = franka.add_body()
        franka.add_joint_fixed(-1, body_table)
        franka.add_shape_box(body_table, xform=wp.transform(p=pos, q=rot), hx=0.4, hy=0.8, hz=0.15)

        # MESH (carrot)
        [mesh_vertices, mesh_indices] = load_mesh(newton.examples.get_asset("fruits/SM_CarrotA_Carrot_0/SM_CarrotA_Carrot_0.obj"))
        mesh_carrot = newton.Mesh(mesh_vertices, mesh_indices)
        body_mesh_carrot = franka.add_body(xform=wp.transform(p=wp.vec3(0.55, 0.45, 0.55)))
        franka.add_joint_free(body_mesh_carrot)
        franka.add_shape_mesh(body_mesh_carrot, mesh = mesh_carrot, cfg=newton.ModelBuilder.ShapeConfig(density=710.0))

        # MESH (bellpepper)
        [mesh_vertices, mesh_indices] = load_mesh(newton.examples.get_asset("fruits/SM_KingOysterMushroom_KingOysterMushroom_0/SM_KingOysterMushroom_KingOysterMushroom_0.obj"))
        mesh_bellpepper = newton.Mesh(mesh_vertices, mesh_indices)
        body_mesh_bellpepper = franka.add_body(xform=wp.transform(p=wp.vec3(0.6, 0.3, 0.45)))
        franka.add_joint_free(body_mesh_bellpepper)
        franka.add_shape_mesh(body_mesh_bellpepper, mesh = mesh_bellpepper, cfg=newton.ModelBuilder.ShapeConfig(density=710.0))

        # MESH (broccoli)
        [mesh_vertices, mesh_indices] = load_mesh(newton.examples.get_asset("fruits/SM_Eggplant_Eggplant_0/SM_Eggplant_Eggplant_0.obj"))
        mesh_broccoli = newton.Mesh(mesh_vertices, mesh_indices)
        body_mesh_broccoli = franka.add_body(xform=wp.transform(p=wp.vec3(0.65, -0.3, 0.45)))
        franka.add_joint_free(body_mesh_broccoli)
        franka.add_shape_mesh(body_mesh_broccoli, mesh = mesh_broccoli, cfg=newton.ModelBuilder.ShapeConfig(density=710.0))

        # # Add a box
        # pos = wp.vec3(0.6, 0.0, 0.43)
        # rot = wp.quat_identity()
        # body_box = franka.add_body(xform=wp.transform(p=pos, q=rot))
        # franka.add_joint_free(body_box)
        # franka.add_shape_box(body_box, hx=0.03, hy=0.03, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))

        # # Set friction
        for i in range(len(franka.shape_material_mu)):
            franka.shape_material_mu[i] = 1.0
            franka.shape_material_kd[i] = 1.0e8

        # ------------------------------------------------------------------
        # Finalization and initialization of computational components
        # ------------------------------------------------------------------

        # Finalize builder
        self.model = franka.finalize()
        self.model.ground = True

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
            njmax=50000, # large enough to avoid nefc overflow
            ncon_per_env=50000, # large enough to avoid illegal mem access
            solver='newton',
            cone="elliptic",
            use_mujoco_cpu=self.use_mujoco_cpu, # mujoco-cpu or mujoco-warp
            # use_mujoco_contacts=True, # incorrect collision when using mujoco-warp
            # use_mujoco_contacts=False, # incorrect friction when using mujoco-warp
            contact_stiffness_time_const=self.sim_dt # important param to ensure zero penetration
        )

        self.capture()

        # self.usd_viewer = newton.viewer.ViewerUSD(output_path="fruits.usd")
        # self.usd_viewer.set_model(self.model)

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
        if wp.get_device().is_cuda and not self.use_mujoco_cpu:
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

            # set control in supsteps
            # self.update_control()

            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

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
        ik_joint_q_np = ik_joint_q.numpy()


        # Position control for [gripper]
        joint_limit_lower_np = self.model.joint_limit_lower.numpy()
        joint_limit_upper_np = self.model.joint_limit_upper.numpy()
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            if self.open_left_gripper:
                ik_joint_q_np[self.left_gripper_joint_indices] = joint_limit_upper_np[self.left_gripper_joint_indices]
            else:
                ik_joint_q_np[self.left_gripper_joint_indices] = joint_limit_lower_np[self.left_gripper_joint_indices]

            if self.open_right_gripper:
                ik_joint_q_np[self.right_gripper_joint_indices] = joint_limit_upper_np[self.right_gripper_joint_indices]
            else:
                ik_joint_q_np[self.right_gripper_joint_indices] = joint_limit_lower_np[self.right_gripper_joint_indices]

        ik_joint_q.assign(ik_joint_q_np)

        # Align the self.state with the target body in the viewer
        newton.eval_fk(self.model, ik_joint_q, self.model.joint_qd, self.state)
        
        # Limit the joint movement in one frame
        move, target = limit_joint_move(ik_joint_q.numpy(), self.state_0.joint_q.numpy(), 20.0, self.frame_dt)

        # Set target q control for [controllable joint] and [gripper]
        joint_target_np = self.control.joint_target.numpy()
        joint_target_np[self.controllable_joint_indices] = target.flatten()[self.controllable_joint_indices]
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            joint_target_np[self.left_gripper_joint_indices]=(target.flatten()[self.left_gripper_joint_indices])
            joint_target_np[self.right_gripper_joint_indices]=(target.flatten()[self.right_gripper_joint_indices])
        self.control.joint_target.assign(joint_target_np)

        # Set joint qd for [controllable joint] and [gripper]
        joint_qd_np = self.state_0.joint_qd.numpy()
        ik_joint_qd_np = move/self.frame_dt
        joint_qd_np[self.controllable_joint_indices] = ik_joint_qd_np[self.controllable_joint_indices]
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:
            joint_qd_np[self.left_gripper_joint_indices]=(ik_joint_qd_np[self.left_gripper_joint_indices])
            joint_qd_np[self.right_gripper_joint_indices]=(ik_joint_qd_np[self.right_gripper_joint_indices])
        self.ik_joint_qd.assign(joint_qd_np)
        self.state_0.joint_qd.assign(joint_qd_np)

        # Set joint velocity for the grippers
        gripper_vel = 0.2

        # Velocity control for [gripper]
        if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
            joint_target_np = self.control.joint_target.numpy()

            vel = linear_map(self.open_left_gripper, -gripper_vel, gripper_vel)
            joint_target_np[self.left_gripper_joint_indices] = vel
            vel = linear_map(self.open_right_gripper, -gripper_vel, gripper_vel)
            joint_target_np[self.right_gripper_joint_indices] = vel
            self.control.joint_target.assign(joint_target_np)


        # Physics step
        if self.physics_graph:
            wp.capture_launch(self.physics_graph)
        else:
            self.physics_simulate()

        self.sim_time += self.frame_dt
        self.sim_frame += 1

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
        if self.use_dump_image:
            io_util.dump_gl_frame_image(self.viewer.renderer._screen_width,self.viewer.renderer._screen_height,f"img_{self.sim_frame}.png")

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # parser.set_defaults(viewer="usd", output_path="lift2_interactive_control.usd")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
