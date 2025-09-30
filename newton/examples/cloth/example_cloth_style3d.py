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
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.utils
from newton import Mesh, ParticleFlags
from newton._src.solvers.style3d import Collision


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # must be an even number when using CUDA Graph
        self.sim_substeps = 10
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4

        self.viewer = viewer
        builder = newton.Style3DModelBuilder(up_axis=newton.Axis.Z)

        use_cloth_mesh = True
        if use_cloth_mesh:
            asset_path = newton.utils.download_asset("style3d")

            # Garment
            # garment_usd_name = "Women_Skirt"
            # garment_usd_name = "Female_T_Shirt"
            garment_usd_name = "Women_Sweatshirt"

            usd_stage = Usd.Stage.Open(str(asset_path / "garments" / (garment_usd_name + ".usd")))
            usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath(str("/Root/" + garment_usd_name + "/Root_Garment")))

            garment_prim = UsdGeom.PrimvarsAPI(usd_geom_garment.GetPrim()).GetPrimvar("st")
            garment_mesh_indices = np.array(usd_geom_garment.GetFaceVertexIndicesAttr().Get())
            garment_mesh_points = np.array(usd_geom_garment.GetPointsAttr().Get())
            garment_mesh_uv_indices = np.array(garment_prim.GetIndices())
            garment_mesh_uv = np.array(garment_prim.Get()) * 1e-3

            # Avatar
            usd_stage = Usd.Stage.Open(str(asset_path / "avatars" / "Female.usd"))
            usd_geom_avatar = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2"))
            avatar_mesh_indices = np.array(usd_geom_avatar.GetFaceVertexIndicesAttr().Get())
            avatar_mesh_points = np.array(usd_geom_avatar.GetPointsAttr().Get())

            builder.add_aniso_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
                panel_verts=garment_mesh_uv.tolist(),
                panel_indices=garment_mesh_uv_indices.tolist(),
                vertices=garment_mesh_points.tolist(),
                indices=garment_mesh_indices.tolist(),
                density=0.3,
                scale=1.0,
                particle_radius=5.0e-3,
            )
            builder.add_shape_mesh(
                body=builder.add_body(),
                xform=wp.transform(
                    p=wp.vec3(0, 0, 0),
                    q=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
                ),
                mesh=Mesh(avatar_mesh_points, avatar_mesh_indices),
            )
            # fixed_points = [0]
            fixed_points = []
        else:
            grid_dim = 100
            grid_width = 1.0
            cloth_density = 0.3
            builder.add_aniso_cloth_grid(
                pos=wp.vec3(-0.5, 0.0, 2.0),
                rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
                dim_x=grid_dim,
                dim_y=grid_dim,
                cell_x=grid_width / grid_dim,
                cell_y=grid_width / grid_dim,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=cloth_density * (grid_width * grid_width) / (grid_dim * grid_dim),
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                tri_ka=1.0e2,
                tri_kd=2.0e-6,
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )
            fixed_points = [0, grid_dim]

        # add a table
        builder.add_ground_plane()
        self.model = builder.finalize()

        # set fixed points
        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_points:
            flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2e-2
        self.model.soft_contact_margin = 0.35e-2
        self.model.soft_contact_ke = 1.0e1
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
            collision_handler=Collision(self.model),
            # collision_handler = CollisionHandler(self.model),
        )
        self.solver.precompute(
            builder,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example)
