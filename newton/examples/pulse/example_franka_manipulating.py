###########################################################################
# 
###########################################################################

# WH:
import numpy as np

import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

import os


def allclose(a: wp.vec3, b: wp.vec3, rtol=1e-5, atol=1e-8):
    return (
        wp.abs(a[0] - b[0]) <= (atol + rtol * wp.abs(b[0]))
        and wp.abs(a[1] - b[1]) <= (atol + rtol * wp.abs(b[1]))
        and wp.abs(a[2] - b[2]) <= (atol + rtol * wp.abs(b[2]))
    )

def vec_rotation(x: float, y: float, z: float) -> wp.transform:
    """Convert plane coordinates given by the plane normal and its offset along the normal to a transform."""
    normal = wp.normalize(wp.vec3(x, y, z))
    if allclose(normal, wp.vec3(0.0, 0.0, 1.0)):
        # no rotation necessary
        return wp.quat(0.0, 0.0, 0.0, 1.0)
    elif allclose(normal, wp.vec3(0.0, 0.0, -1.0)):
        # 180 degree rotation around x-axis
        return wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        c = wp.cross(wp.vec3(0.0, 0.0, 1.0), normal)
        angle = wp.asin(wp.length(c))
        # adjust for arcsin ambiguity
        if wp.dot(normal, wp.vec3(0.0, 0.0, 1.0)) < 0:
            angle = wp.pi - angle
        axis = c / wp.length(c)
        return wp.quat_from_axis_angle(axis, angle)

@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_env: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    env_id = wp.tid()
    tf = body_q[bodies_per_env * env_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[env_id] = wp.spatial_vector(ang_diff[0], ang_diff[1], ang_diff[2], pos_diff[0], pos_diff[1], pos_diff[2])

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

        franka.default_shape_cfg.density = 100.0

        # NOTE:
        # The original Franka model from Newton has issues with missing collisions and incorrect coloring when used with the MuJoCo solver.
        # As a temporary workaround, the Genesis Franka model is used instead.
        # Steps:
        #   1. Download the Genesis Franka model from:
        #      http://69.235.177.182:8081/externalrepo/Genesis/-/tree/9b8a40a84cf74b3fd5d63ec42877c98192ba681c/genesis/assets/xml/franka_emika_panda
        #   2. Move the downloaded model to: ./newton/examples/assets/franka_emika_panda

        # franka.add_urdf(
        #     newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        #     floating=False,
        #     enable_self_collisions=False,
        #     collapse_fixed_joints=True,
        #     # force_show_colliders=False,
        # )

        franka.add_mjcf(
            newton.examples.get_asset("franka_emika_panda/panda.xml"),
            # xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            # force_show_colliders=False,
        )

        # Set initial joint configuration
        franka.joint_q[:7] = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, -0.7]

        self.robot_joint_q_cnt = len(franka.joint_q)

        # Add a box
        pos = wp.vec3(0.5, 0.0, 0.13)
        rot = wp.quat_identity()
        body_box = franka.add_body(xform=wp.transform(p=pos, q=rot))
        franka.add_joint_free(body_box)
        franka.add_shape_box(body_box, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=1.0))

        # Update friction and material properties for all shapes
        for i in range(len(franka.shape_material_mu)):
            # print(f"shape {i}, mu={franka.shape_material_mu[i]}, ka={franka.shape_material_ka[i]}")
            franka.shape_material_mu[i] = 1.0
            franka.shape_material_ka[i] = 0.002
            franka.shape_is_solid[i] = True

        # Add the ground
        franka.add_ground_plane()

        self.graph = None
        self.model = franka.finalize()
        self.model.ground = True

        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        # Target state initialization
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)


        # ------------------------------------------------------------------
        # End effector
        # ------------------------------------------------------------------
        # self.ee_index = 10  # For uncollapsed URDF from Newton asset
        # self.ee_index = 6   # For collapsed URDF from Newton asset
        # 12, 13 are the two fingers
        
        self.ee_index = 8  # For panda.xml (Genesis MJCF)

        # Persistent gizmo transform (pass-by-ref mutated by viewer)
        body_q_np = self.state_0.body_q.numpy()
        print('body_q_np',body_q_np.shape)
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])


        # Control setup
        # [ p.xyz, rot.wxyz, gripper ]
        self.endeffector_id = self.ee_index
        self.endeffector_offset = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())
        print('ee_tf=',self.ee_tf)
        self.target = [*wp.transform_get_translation(self.ee_tf), *wp.transform_get_rotation(self.ee_tf)]

        self.open_gripper = 1 # 1 for open, 0 for close

        # # ------------------------------------------------------------------
        # # IK setup (single problem, single EE)
        # # ------------------------------------------------------------------
        # # residual layout:
        # # [0..2]  : position (3)
        # # [3..5]  : rotation (3)
        # # [6..]   : joint limits (joint_coord_count)
        # total_residuals = 6 + self.model.joint_coord_count

        # def _q2v4(q):
        #     return wp.vec4(q[0], q[1], q[2], q[3])

        # # Position objective
        # self.pos_obj = ik.IKPositionObjective(
        #     link_index=self.ee_index,
        #     link_offset=wp.vec3(0.0, 0.0, 0.0),
        #     target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
        #     n_problems=1,
        #     total_residuals=total_residuals,
        #     residual_offset=0,
        # )

        # # Rotation objective
        # self.rot_obj = ik.IKRotationObjective(
        #     link_index=self.ee_index,
        #     link_offset_rotation=wp.quat_identity(),
        #     target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        #     n_problems=1,
        #     total_residuals=total_residuals,
        #     residual_offset=3,
        # )

        # # Joint limit objective
        # self.obj_joint_limits = ik.IKJointLimitObjective(
        #     joint_limit_lower=self.model.joint_limit_lower,
        #     joint_limit_upper=self.model.joint_limit_upper,
        #     n_problems=1,
        #     total_residuals=total_residuals,
        #     residual_offset=6,
        #     weight=10.0,
        # )

        # # Variables the solver will update
        # self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        # print('joint_q',self.joint_q.shape)

        # self.ik_iters = 24
        # self.solver = ik.IKSolver(
        #     model=self.model,
        #     joint_q=self.joint_q,
        #     objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
        #     lambda_initial=0.1,
        #     jacobian_mode=ik.IKJacobianMode.MIXED,
        # )

        # WH: add rigid solver
        # self.rigid_solver = newton.solvers.SolverFeatherstone(self.model)
        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            # nefc_per_env=500,
            njmax=500,
            ncon_per_env=500,
            solver='newton',
            impratio=100,
        )
        print('set up mujoco')
        self.set_up_control()

        self.capture()



    # ----------------------------------------------------------------------
    # WH: from manipulating cube
    # ----------------------------------------------------------------------
    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        # in_dim = self.model.joint_dof_count
        in_dim = self.robot_joint_q_cnt

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        # for robot control
        # self.delta_q = wp.empty(self.model.joint_count, dtype=float)
        self.joint_q_des = wp.array(self.model.joint_q.numpy(), dtype=float)

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = newton.utils.transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()[:in_dim]


    def compute_body_jacobian(
        self,
        model: newton.Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        # in_dim = model.joint_dof_count
        in_dim = self.robot_joint_q_cnt
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            newton.eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad[ : in_dim])
            tape.zero()

    def generate_control_joint_qd(
        self,
        state_in: newton.State,
    ):
        include_rotation = True
        self.target = [*wp.transform_get_translation(self.ee_tf), *wp.transform_get_rotation(self.ee_tf)]

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.model.body_count,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        
        # J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        J = self.J_flat.numpy().reshape(-1, self.robot_joint_q_cnt)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        # 2. Compute null-space projector
        #    I is size [num_joints x num_joints]
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()[:self.robot_joint_q_cnt]

        # 3. Define a desired "elbow-up" reference posture
        #    (For example, one that keeps joint 2 or 3 above a certain angle.)
        #    Adjust indices and angles to your robot's kinematics.
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]  # e.g., set elbow joint around 1 rad to keep it up

        # 4. Define a null-space velocity term pulling joints toward q_des
        #    K_null is a small gain so it doesn't override main task
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        # 5. Combine primary task and null-space controller
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # TODO: Apply gripper finger control
        gripper_target = ((self.open_gripper) * 0.2)
        delta_q[-2] = (gripper_target - q[-2])
        delta_q[-1] = (gripper_target - q[-1])

        self.target_joint_qd.assign(delta_q)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.state_0.joint_qd[:self.robot_joint_q_cnt].assign(self.target_joint_qd[:self.robot_joint_q_cnt])

            # self.rigid_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.rigid_solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)


            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)



        # self.solver.solve(iterations=self.ik_iters)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        self.pos_obj.set_target_position(0, wp.transform_get_translation(self.ee_tf))
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("e"):
                self.open_gripper = 0
            else:
                self.open_gripper = 1

        # TODO:
        # self._push_targets_from_gizmos()
        self.generate_control_joint_qd(self.state_0)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        # print("CURRENT Q ", self.state_0.body_q.numpy())
        # print("CURRENT QD", self.state_0.body_qd.numpy())
        # print("CURRENT JQ", self.state_0.joint_q.numpy())
        # print("MJW CONDIM",self.rigid_solver.mjw_model.geom_condim.numpy())

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmo (viewer will draw & mutate transform in-place)
        self.viewer.log_gizmo("target_tcp", self.ee_tf)

        # Visualize the current articulated state
        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        # self.viewer.log_state(self.state)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.end_frame()
        wp.synchronize()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)