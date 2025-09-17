import warp as wp
import numpy as np
from pxr import Usd, UsdGeom

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
        self.iterations = 7
        self.sim_dt = self.frame_dt / self.sim_substeps

        #   contact
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

        self.scene = newton.ModelBuilder()
        self.soft_contact_max = 1000000

        # NOTE: Set gripper control mode
        # self.gripper_control_type = GripperControlType.TARGET_VELOCITY
        self.gripper_control_type = GripperControlType.TARGET_POSITION

        # NOTE: Enable force feedback from particle–body contacts
        self.use_particle_body_force = True

        self.viewer = viewer

        # Add the robot
        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(
                (-0.5, -0.5, -0.1),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,
        )
        self.robot_joint_q_cnt = len(franka.joint_q)

        # Configure target position control for arm joints.
        for i in range(self.robot_joint_q_cnt):
            franka.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            franka.joint_target_ke[i] = 3000.0
            franka.joint_target_kd[i] = 10.0

        # Configure joint limits for finger joints
        for i in range(self.robot_joint_q_cnt-2, self.robot_joint_q_cnt):

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
                franka.joint_target_kd[i] = 1.0

        franka.joint_q[:7] = [
            0, -0.25, 0, -1.5,
            0, 1.2, -0.8
        ]

        # Define the desired initial end-effector pose.
        # This pose will be used with IK later to initialize the robot state.
        init_ee_tf = wp.transform(
            wp.vec3(-0.075, -0.5, 0.7),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi * 0.5))


        # Add to self.scene
        xform = wp.transform(wp.vec3(0), wp.quat_identity())
        self.scene.add_builder(franka, xform)
        self.bodies_per_env = franka.body_count
        self.dof_q_per_env = franka.joint_coord_count
        self.dof_qd_per_env = franka.joint_dof_count

        # Set indices for the end-effector and fingers
        self.ee_index = 10  # hardcoded for now, end effector body index
        self.lf_index = self.robot_joint_q_cnt - 2 # left finger joint index
        self.rf_index = self.robot_joint_q_cnt - 1 # right finger joint index

        # Add a table
        pos = wp.vec3(0.0, -0.5, 0.1)
        rot = wp.quat_identity()
        body_box = self.scene.add_body()
        self.scene.add_joint_fixed(-1, body_box)
        self.scene.add_shape_box(body_box, xform=wp.transform(p=pos, q=rot), hx=0.4, hy=0.4, hz=0.1)

        # # Add the T-shirt
        # usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        # usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        # mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        # mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        # vertices = [wp.vec3(v) for v in mesh_points]
        # self.scene.add_cloth_mesh(
        #     vertices=vertices,
        #     indices=mesh_indices,
        #     rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
        #     pos=wp.vec3(0.0, 0.70, 0.28),
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     density=0.2,
        #     scale=0.01,
        #     tri_ke=self.tri_ke,
        #     tri_ka=self.tri_ka,
        #     tri_kd=self.tri_kd,
        #     edge_ke=self.bending_ke,
        #     edge_kd=self.bending_kd,
        #     particle_radius=self.cloth_particle_radius,
        # )

        # Add a square cloth
        cloth_res=[24, 24]
        cloth_dx=[0.005, 0.005]
        cloth_mass = 0.2 * cloth_dx[0] * cloth_dx[1] * cloth_res[0] * cloth_res[1]
        cloth_len = [cloth_res[i]*cloth_dx[i] for i in range(2)]
        print("cloth mass=", cloth_mass)
        common_params = {
            "pos": wp.vec3(0.0, -0.5, 0.7),
            "rot": wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5),
            # "rot": wp.quat_identity(),
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "dim_x": cloth_res[0],
            "dim_y": cloth_res[1],
            "cell_x": cloth_dx[0],
            "cell_y": cloth_dx[1],
            "mass": cloth_mass,
            "fix_top": True,
            "edge_ke": self.bending_ke,
            "edge_kd": self.bending_kd,
            "particle_radius": self.cloth_particle_radius,
        }
        solver_params = {
            "tri_ke": self.tri_ke,
            "tri_ka": self.tri_ka,
            "tri_kd": self.tri_kd,
        }
        self.scene.add_cloth_grid(**common_params, **solver_params)

        self.scene.color()

        # Add the ground
        self.scene.add_ground_plane()


        # Warp compute graphs
        self.ik_graph = None
        self.physics_graph = None

        # Finialize builder
        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            pos = type(self.viewer.camera.pos)(2.0, -0.5, 1.4)
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
        self.open_gripper = 1

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = init_ee_tf
        self.old_ee_tf = init_ee_tf

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
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.ik_iters = 24

        # IK solver
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.ik_joint_q,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        # Rigid body solver
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=500,
            solver='newton',
            cone="elliptic",
            contact_stiffness_time_const=self.sim_dt # important param to ensure zero penetration
        )

        # Cloth solver
        self.model.edge_rest_angle.zero_()
        self.cloth_solver = newton.solvers.SolverVBDPulse(
            self.model,
            iterations=self.iterations,
            self_contact_radius=self.self_contact_radius,
            self_contact_margin=self.self_contact_margin,
            handle_self_contact=True,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
            integrate_with_external_rigid_solver=True,
            collision_detection_interval=-1,
        )

        self.capture()

        # Assign IK solution to the state
        # Run the IK solver once
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()
        # Forward kinematics to get the body poses
        ik_joint_q = self.ik_joint_q.flatten()
        newton.eval_fk(self.model, ik_joint_q, self.model.joint_qd, self.state)
        # Assign IK joint/body state
        self.state.joint_q.assign(ik_joint_q)
        self.state.joint_qd.zero_()
        self.state.body_qd.zero_()
        for state in [self.state_0, self.state_1]:
            # Assign IK joint/body state
            state.joint_q.assign(ik_joint_q)
            state.joint_qd.zero_()
            state.body_q.assign(self.state.body_q)
            state.body_qd.zero_()
            # Assign init particle state
            state.particle_q.assign(self.state.particle_q)
            state.particle_qd.zero_()
            state.particle_f.zero_()


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
        # Rebuild cloth
        self.cloth_solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            # Collision detection
            self.contacts = self.model.collide(self.state_0)

            # TODO: comment out this line to apply particle-body contact force onto rigid bodies 
            if not self.use_particle_body_force:
                self.state_0.clear_forces()
            # self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Clear particle info for rigid_solver
            particle_count = self.model.particle_count
            self.model.particle_count = 0

            # Solve the rigid bodies
            self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Recover the particle info
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            # Solve the cloth, add force onto state_1
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=self.cloth_body_contact_margin)
            self.cloth_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # Swap state
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        self.pos_obj.set_target_position(0, wp.transform_get_translation(self.ee_tf))
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

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

        # IK step, update self.ik_joint_q as the target pose
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_simulate()

        ik_joint_q = self.ik_joint_q.flatten()

        # Align the self.state with the target body in the viewer
        newton.eval_fk(self.model, ik_joint_q, self.model.joint_qd, self.state)

        # Limit the joint movement for the arm in one frame
        move, target = limit_joint_move(ik_joint_q[0:self.lf_index].numpy(), self.state_0.joint_q[0:self.lf_index].numpy(), 2.0, self.frame_dt)

        # Set joint target position for the arm
        self.control.joint_target[0:self.lf_index].assign(target.flatten()[0:self.lf_index])

        # Manually reset joint velocity to stablize the simulation (Not sure why)
        self.state_0.joint_qd[0:self.lf_index].assign(move[0:self.lf_index]/self.frame_dt)

        # Set joint target position for the fingers
        # self.control.joint_target[self.lf_index:self.lf_index+1].assign([
        #     linear_map(self.open_gripper, 0.02, 0.04)
        # ])
        # self.control.joint_target[self.rf_index:self.rf_index+1].assign([
        #     linear_map(self.open_gripper, 0.02, 0.04),
        # ])

        # Set joint target for the fingers
        gripper_vel = 0.2

        # Position control
        if self.gripper_control_type == GripperControlType.TARGET_POSITION:

            # Get IK target
            ik_lf_q = self.model.joint_limit_lower[self.lf_index:self.lf_index+1]
            ik_rf_q = self.model.joint_limit_lower[self.rf_index:self.rf_index+1]
            if self.open_gripper:
                ik_lf_q = self.model.joint_limit_upper[self.lf_index:self.lf_index+1]
                ik_rf_q = self.model.joint_limit_upper[self.rf_index:self.rf_index+1]

            ik_joint_q[self.lf_index:self.lf_index+1].assign(ik_lf_q)
            ik_joint_q[self.rf_index:self.rf_index+1].assign(ik_rf_q)

            # Get control signal
            move, target = limit_joint_move(ik_joint_q[self.lf_index:self.rf_index+1].numpy(), self.state_0.joint_q[self.lf_index:self.rf_index+1].numpy(), gripper_vel, self.frame_dt)

            # Set control signal
            self.control.joint_target[self.lf_index:self.rf_index+1].assign(target.flatten())
            self.state_0.joint_qd[self.lf_index:self.rf_index+1].assign(move/self.frame_dt)

        # Velocity control
        if self.gripper_control_type == GripperControlType.TARGET_VELOCITY:
            self.control.joint_target[self.lf_index:self.lf_index+1].assign([
                linear_map(self.open_gripper, -gripper_vel, gripper_vel)
            ])
            self.control.joint_target[self.rf_index:self.rf_index+1].assign([
                linear_map(self.open_gripper, -gripper_vel, gripper_vel),
            ])

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
        self.viewer.log_state(self.state_0)

        # Visualize contacts
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Visualize particle-body contact forces on rigid bodies
        self.viewer.log_particle_body_contacts(self.contacts, self.state_0)

        # Visualize particle forces on cloth
        self.viewer.log_contact_forces(self.cloth_solver.particle_forces, self.state_0)
        self.viewer.end_frame()

        wp.synchronize()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
