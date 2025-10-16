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
# Example MPM ANYmal
#
# Shows ANYmal C with a pretrained policy coupled with implicit MPM sand.
#
# Example usage (via unified runner):
#   python -m newton.examples mpm_anymal --viewer gl
###########################################################################

import sys

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot.example_robot_anymal_c_walk import compute_obs, lab_to_mujoco, mujoco_to_lab
from newton.solvers import SolverImplicitMPM


@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    res_mesh: wp.uint64,
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
):
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    X_wb = body_q[shape_body_id[shape_id]]

    cur_p = res.points[v] + dt * res.velocities[v]
    next_p = wp.transform_point(X_wb, p)
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p


class Example:
    def __init__(
        self,
        viewer,
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        sand_friction=0.48,
    ):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.device = wp.get_device()

        # import the robot model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        # set initial joint positions
        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for key, value in initial_q.items():
            builder.joint_q[builder.joint_key.index(key) + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        # add sand particles
        density = 2500.0
        particle_lo = np.array([-0.5, -0.5, 0.0])  # emission lower bound
        particle_hi = np.array([0.5, 2.5, 0.15])  # emission upper bound
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(builder, particle_res, particle_lo, particle_hi, density)

        # finalize model
        self.model = builder.finalize()

        self.model.particle_mu = sand_friction
        self.model.particle_ke = 1.0e15

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50

        # global defaults
        mpm_options.hardening = 0.0
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0

        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        # Select and merge meshes for robot/sand collisions
        collider_body_idx = [idx for idx, key in enumerate(builder.body_key) if "SHANK" in key]
        collider_shape_ids = np.concatenate(
            [[m for m in self.model.body_shapes[b] if self.model.shape_source[m]] for b in collider_body_idx]
        )

        collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
            [self.model.shape_source[m].vertices for m in collider_shape_ids],
            [self.model.shape_source[m].indices for m in collider_shape_ids],
            [self.model.shape_scale.numpy()[m] for m in collider_shape_ids],
            collider_shape_ids,
        )

        self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        self.collider_rest_points = collider_points
        self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        mpm_model.setup_collider([self.collider_mesh], collider_friction=[0.5], collider_adhesion=[0.0])

        # setup solvers
        self.solver = newton.solvers.SolverMuJoCo(self.model, ls_parallel=True, njmax=50)
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._update_collider_mesh(self.state_0)

        # Setup control policy
        self.control = self.model.control()

        q0 = wp.to_torch(self.state_0.joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        # Download the policy from the newton-assets repository
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

        # Pre-compute tensors that don't change during simulation
        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))], device=self.torch_device
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)

        self._auto_forward = True

        # set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

    def apply_control(self):
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.torch_device,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.gather(self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0))
            a = self.joint_pos_initial + 0.5 * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            # copy action targets to control buffer
            wp.copy(self.control.joint_target, a_wp)

    def simulate_robot(self):
        # robot substeps
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        # sand step (in-place on frame dt)
        self._update_collider_mesh(self.state_0)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                # disable forward motion
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1

        # compute control before graph/step
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        # MPM solver step is not graph-capturable yet
        self.simulate_sand()

        self.sim_time += self.frame_dt

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 0.9,  # This threshold assumes 100 frames
        )

        forward_vel_min = wp.spatial_vector(-0.2, 0.9, -0.2, -0.8, -0.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.2, 1.1, 0.2, 0.8, 0.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.utils.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.03,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _update_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_rest_points,
                self.collider_shape_ids,
                self.collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.collider_mesh.refit()


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)

    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    rng = np.random.default_rng()
    points += 2.0 * radius * (rng.random(points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel

    builder.particle_mass = np.full(points.shape[0], mass)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.ones(points.shape[0], dtype=int)


def _merge_meshes(
    points: list[np.array],
    indices: list[np.array],
    scales: list[np.array],
    shape_ids: list[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales, strict=False)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.48)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # This example requires a GPU device
    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # Create example and load policy
    example = Example(
        viewer,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        tolerance=args.tolerance,
        sand_friction=args.sand_friction,
    )

    # Run via unified example runner
    newton.examples.run(example, args)
