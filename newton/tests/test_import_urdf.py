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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import newton
import newton.examples
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import assert_np_equal

MESH_URDF = """
<robot name="mesh_test">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="{filename}" scale="1.0 1.0 1.0"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="0 0 0"/>
        </visual>
    </link>
</robot>
"""

MESH_OBJ = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

# Front face
f 1 2 3
f 1 3 4
# Back face
f 5 7 6
f 5 8 7
# Right face
f 2 6 7
f 2 7 3
# Left face
f 1 4 8
f 1 8 5
# Top face
f 4 3 7
f 4 7 8
# Bottom face
f 1 5 6
f 1 6 2
"""

INERTIAL_URDF = """
<robot name="inertial_test">
    <link name="base_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0"
                     iyy="1.0" iyz="0.0"
                     izz="1.0"/>
        </inertial>
        <visual>
            <geometry>
                <capsule radius="0.5" length="1.0"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="1.5707963 0 0"/>
        </visual>
    </link>
</robot>
"""

SPHERE_URDF = """
<robot name="sphere_test">
    <link name="base_link">
        <visual>
            <geometry>
                <sphere radius="0.5"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="0 0 0"/>
        </visual>
    </link>
</robot>
"""

SELF_COLLISION_URDF = """
<robot name="self_collision_test">
    <link name="base_link">
        <collision>
            <geometry><sphere radius="0.5"/></geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
    <link name="far_link">
        <collision>
            <geometry><sphere radius="0.5"/></geometry>
            <origin xyz="1.0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""

JOINT_URDF = """
<robot name="joint_test">
<link name="base_link"/>
<link name="child_link"/>
<joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 1.0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.23" upper="3.45"/>
</joint>
</robot>
"""


class TestImportUrdf(unittest.TestCase):
    @staticmethod
    def parse_urdf(urdf: str, builder: newton.ModelBuilder, res_dir: dict[str, str] | None = None, **kwargs):
        """Parse the specified URDF file from a directory of files.
        urdf: URDF file to parse
        res_dir: dict[str, str]: (filename, content): extra resources files to include in the directory"""

        urdf_filename = "robot.urdf"
        # Create a temporary directory to store files
        res_dir = res_dir or {}
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write all files to the temporary directory
            for filename, content in {urdf_filename: urdf, **res_dir}.items():
                file_path = Path(temp_dir) / filename
                with open(file_path, "w") as f:
                    f.write(content)

            # Parse the URDF file
            urdf_path = Path(temp_dir) / urdf_filename
            builder.add_urdf(str(urdf_path), up_axis="Y", **kwargs)

    def test_sphere_urdf(self):
        # load a urdf containing a sphere with r=0.5 and pos=(1.0,2.0,3.0)
        builder = newton.ModelBuilder()
        self.parse_urdf(SPHERE_URDF, builder)

        assert builder.shape_count == 1
        assert builder.shape_type[0] == newton.GeoType.SPHERE
        assert builder.shape_scale[0][0] == 0.5
        assert_np_equal(builder.shape_transform[0][:], np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))

    def test_mesh_urdf(self):
        # load a urdf containing a cube mesh with 8 verts and 12 faces
        for mesh_src in ("file", "http"):
            with self.subTest(mesh_src=mesh_src):
                builder = newton.ModelBuilder()
                if mesh_src == "file":
                    self.parse_urdf(MESH_URDF.format(filename="cube.obj"), builder, {"cube.obj": MESH_OBJ})
                else:

                    def mock_mesh_download(dst, url: str):
                        dst.write(MESH_OBJ.encode("utf-8"))

                    with patch("newton._src.utils.import_urdf._download_file", side_effect=mock_mesh_download):
                        self.parse_urdf(MESH_URDF.format(filename="http://example.com/cube.obj"), builder)

                assert builder.shape_count == 1
                assert builder.shape_type[0] == newton.GeoType.MESH
                assert_np_equal(builder.shape_scale[0], np.array([1.0, 1.0, 1.0]))
                assert_np_equal(builder.shape_transform[0][:], np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
                assert builder.shape_source[0].vertices.shape[0] == 8
                assert builder.shape_source[0].indices.shape[0] == 3 * 12

    def test_inertial_params_urdf(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(INERTIAL_URDF, builder, ignore_inertial_definitions=False)

        assert builder.shape_type[0] == newton.GeoType.CAPSULE
        assert builder.shape_scale[0][0] == 0.5
        assert builder.shape_scale[0][1] == 0.5  # half height
        assert_np_equal(
            np.array(builder.shape_transform[0][:]), np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]), tol=1e-6
        )

        # Check inertial parameters
        assert_np_equal(builder.body_mass[0], np.array([1.0]))
        assert_np_equal(builder.body_inertia[0], np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        assert_np_equal(builder.body_com[0], np.array([0.0, 0.0, 0.0]))

    def test_self_collision_filtering_parameterized(self):
        for self_collisions in [False, True]:
            with self.subTest(enable_self_collisions=self_collisions):
                builder = newton.ModelBuilder()
                self.parse_urdf(SELF_COLLISION_URDF, builder, enable_self_collisions=self_collisions)

                assert builder.shape_count == 2

                # Check if collision filtering is applied correctly based on self_collisions setting
                filter_pair = (0, 1)
                if self_collisions:
                    self.assertNotIn(filter_pair, builder.shape_collision_filter_pairs)
                else:
                    self.assertIn(filter_pair, builder.shape_collision_filter_pairs)

    def test_revolute_joint_urdf(self):
        # Test a simple revolute joint with axis and limits
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_URDF, builder)

        # Check joint was created with correct properties
        assert builder.joint_count == 2  # base joint + revolute
        assert builder.joint_type[-1] == newton.JointType.REVOLUTE

        assert_np_equal(builder.joint_limit_lower[-1], np.array([-1.23]))
        assert_np_equal(builder.joint_limit_upper[-1], np.array([3.45]))
        assert_np_equal(builder.joint_axis[-1], np.array([0.0, 0.0, 1.0]))

    def test_cartpole_urdf(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 123.0
        builder.default_shape_cfg.kd = 456.0
        builder.default_shape_cfg.mu = 789.0
        builder.default_joint_cfg.armature = 42.0
        urdf_filename = newton.examples.get_asset("cartpole.urdf")
        builder.add_urdf(
            urdf_filename,
            floating=False,
        )
        self.assertTrue(all(np.array(builder.shape_material_ke) == 123.0))
        self.assertTrue(all(np.array(builder.shape_material_kd) == 456.0))
        self.assertTrue(all(np.array(builder.shape_material_mu) == 789.0))
        self.assertTrue(all(np.array(builder.joint_armature) == 42.0))
        assert builder.body_count == 4

    def test_cylinder_shapes_preserved(self):
        """Test that cylinder geometries are properly imported as cylinders, not capsules."""
        # Create URDF content with cylinder geometry
        urdf_content = """
<robot name="cylinder_test">
    <link name="base_link">
        <collision>
            <geometry>
                <cylinder radius="0.5" length="2.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
        <visual>
            <geometry>
                <cylinder radius="0.5" length="2.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </visual>
    </link>
    <link name="second_link">
        <collision>
            <geometry>
                <capsule radius="0.3" height="1.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = os.path.join(tmpdir, "cylinder_test.urdf")
            with open(urdf_path, "w") as f:
                f.write(urdf_content)

            builder = newton.ModelBuilder()
            builder.add_urdf(urdf_path)

            # Check shape types
            shape_types = list(builder.shape_type)

            # First shape should be cylinder (collision)
            self.assertEqual(shape_types[0], GeoType.CYLINDER)

            # Second shape should be cylinder (visual)
            self.assertEqual(shape_types[1], GeoType.CYLINDER)

            # Third shape should be capsule
            self.assertEqual(shape_types[2], GeoType.CAPSULE)

            # Check cylinder properties - radius and half_height
            # shape_scale stores (radius, half_height, 0) for cylinders
            shape_scale = builder.shape_scale[0]
            self.assertAlmostEqual(shape_scale[0], 0.5)  # radius
            self.assertAlmostEqual(shape_scale[1], 1.0)  # half_height (length/2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
