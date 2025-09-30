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

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()
        Example.emit_particles(builder, options)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.particle_mu = 0.5
        self.model.particle_ke = 1.0e12
        self.model.particle_kd = 0.0

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.points_per_particle = options.points_per_particle
        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        self.state_0 = mpm_model.state()
        self.state_1 = mpm_model.state()

        # Initialize MPM solver and add supplemental state variables
        self.solver = SolverImplicitMPM(mpm_model, mpm_options)

        # Setup grain rendering

        self.grains = self.solver.sample_render_grains(self.state_0, options.points_per_particle)
        grain_radius = options.voxel_size / (3 * options.points_per_particle)
        self.grain_radii = wp.full(self.grains.size, value=grain_radius, dtype=float, device=self.model.device)
        self.grain_colors = wp.full(
            self.grains.size, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.model.device
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)

            # update grains
            self.solver.update_particle_frames(self.state_0, self.state_1, self.sim_dt)
            self.solver.update_render_grains(self.state_0, self.state_1, self.grains, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "grains", points=self.grains.flatten(), radii=self.grain_radii, colors=self.grain_colors, hidden=False
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, 0.5, 2.0])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        Example._spawn_particles(builder, particle_res, particle_lo, particle_hi, density=2500)

    @staticmethod
    def _spawn_particles(
        builder: newton.ModelBuilder,
        res,
        bounds_lo,
        bounds_hi,
        density,
    ):
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

        rng = np.random.default_rng(42)
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)

        builder.particle_q = points
        builder.particle_qd = vel
        builder.particle_mass = np.full(points.shape[0], mass)
        builder.particle_radius = np.full(points.shape[0], radius)

        builder.particle_flags = np.ones(points.shape[0], dtype=int)


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)
    parser.add_argument("--points-per-particle", "-ppp", type=float, default=8)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example)
