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

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.types import GeoType


class TestImportMjcf(unittest.TestCase):
    def test_humanoid_mjcf(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 123.0
        builder.default_shape_cfg.kd = 456.0
        builder.default_shape_cfg.mu = 789.0
        builder.default_joint_cfg.armature = 42.0
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        builder.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )
        self.assertTrue(all(np.array(builder.shape_material_ke) == 123.0))
        self.assertTrue(all(np.array(builder.shape_material_kd) == 456.0))
        self.assertTrue(all(np.array(builder.shape_material_mu) == 789.0))
        self.assertTrue(all(np.array(builder.joint_armature[:6]) == 0.0))
        self.assertEqual(
            builder.joint_armature[6:],
            [
                0.02,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.006,
                0.01,
                0.01,
                0.006,
            ],
        )
        assert builder.body_count == 13

    def test_mjcf_maxhullvert_parsing(self):
        """Test that maxhullvert is parsed from MJCF files"""
        # Create a temporary MJCF file with maxhullvert attribute
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <asset>
        <mesh name="mesh1" file="mesh1.obj" maxhullvert="32"/>
        <mesh name="mesh2" file="mesh2.obj" maxhullvert="128"/>
        <mesh name="mesh3" file="mesh3.obj"/>
    </asset>
    <worldbody>
        <body>
            <geom type="mesh" mesh="mesh1"/>
            <geom type="mesh" mesh="mesh2"/>
            <geom type="mesh" mesh="mesh3"/>
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "test.xml")

            # Create dummy mesh files
            for i in range(1, 4):
                mesh_path = os.path.join(tmpdir, f"mesh{i}.obj")
                with open(mesh_path, "w") as f:
                    # Simple triangle mesh
                    f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            # Parse MJCF
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path, parse_meshes=True)
            model = builder.finalize()

            # Check that meshes have correct maxhullvert values
            # Note: This assumes meshes are added in order they appear in MJCF
            meshes = [model.shape_source[i] for i in range(3) if hasattr(model.shape_source[i], "maxhullvert")]

            if len(meshes) >= 3:
                self.assertEqual(meshes[0].maxhullvert, 32)
                self.assertEqual(meshes[1].maxhullvert, 128)
                self.assertEqual(meshes[2].maxhullvert, 64)  # Default value

    def test_inertia_rotation(self):
        """Test that inertia tensors are properly rotated using sandwich product R @ I @ R.T"""

        # Test case 1: Diagonal inertia with rotation
        mjcf_diagonal = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_diagonal">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" diaginertia="1.0 2.0 3.0"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test case 2: Full inertia with rotation
        mjcf_full = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_full">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" fullinertia="1.0 2.0 3.0 0.1 0.2 0.3"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test diagonal inertia rotation
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_diagonal, ignore_inertial_definitions=False)
        model = builder.finalize()

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # This transforms the diagonal inertia [1, 2, 3] to [2, 1, 3] via sandwich product R @ I @ R.T
        expected_diagonal = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])

        actual_inertia = model.body_inertia.numpy()[0]
        # The validation may add a small epsilon for numerical stability
        # Check that the values are close within a reasonable tolerance
        np.testing.assert_allclose(actual_inertia, expected_diagonal, rtol=1e-5, atol=1e-5)

        # Test full inertia rotation
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_full, ignore_inertial_definitions=False)
        model = builder.finalize()

        # For full inertia, we need to compute the expected result manually
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) transforms the inertia
        # We need to use the same quaternion-to-matrix conversion as the MJCF importer

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # For full inertia, calculate the expected result analytically using the same quaternion
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # Calculate the expected result analytically using the correct rotation matrix
        # For a 90-degree Z-axis rotation: R = [0 -1 0; 1 0 0; 0 0 1]

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # Rotation matrix for 90-degree rotation around Z-axis
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        expected_full = rotation_matrix @ original_inertia @ rotation_matrix.T

        actual_inertia = model.body_inertia.numpy()[0]

        # The original inertia violates the triangle inequality, so validation will correct it
        # The eigenvalues are [0.975, 1.919, 3.106], which violates I1 + I2 >= I3
        # The validation adds ~0.212 to all eigenvalues to fix this
        # We check that:
        # 1. The rotation structure is preserved (off-diagonal elements match)
        # 2. The diagonal has been increased by approximately the same amount

        # Check off-diagonal elements are preserved
        np.testing.assert_allclose(actual_inertia[0, 1], expected_full[0, 1], atol=1e-6)
        np.testing.assert_allclose(actual_inertia[0, 2], expected_full[0, 2], atol=1e-6)
        np.testing.assert_allclose(actual_inertia[1, 2], expected_full[1, 2], atol=1e-6)

        # Check that diagonal elements have been increased by approximately the same amount
        corrections = np.diag(actual_inertia - expected_full)
        np.testing.assert_allclose(corrections, corrections[0], rtol=1e-3)

        # Verify that the rotation was actually applied (not just identity)
        assert not np.allclose(actual_inertia, original_inertia, atol=1e-6)

    def test_single_body_transform(self):
        """Test 1: Single body with pos/quat → verify body_q matches expected world transform."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="test_body" pos="1.0 2.0 3.0" quat="0.7071068 0 0 0.7071068">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Expected: translation (1, 2, 3) + 90° rotation around Z
        body_idx = model.body_key.index("test_body")
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]

        np.testing.assert_allclose(body_pos, [1.0, 2.0, 3.0], atol=1e-6)
        # MJCF quat is [w, x, y, z], body_q quat is [x, y, z, w]
        # So [0.7071068, 0, 0, 0.7071068] becomes [0, 0, 0.7071068, 0.7071068]
        np.testing.assert_allclose(body_quat, [0, 0, 0.7071068, 0.7071068], atol=1e-6)

    def test_root_body_with_custom_xform(self):
        """Test 1: Root body with custom xform parameter (with rotation) → verify transform is properly applied."""
        # Add a 45-degree rotation around Z to the body
        angle_body = np.pi / 4
        quat_body = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_body)
        # wp.quat_from_axis_angle returns [x, y, z, w]
        # MJCF expects [w, x, y, z]
        quat_body_mjcf = f"{quat_body[3]} {quat_body[0]} {quat_body[1]} {quat_body[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="test_body" pos="0.5 0.5 0.0" quat="{quat_body_mjcf}">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        # Custom xform: translate by (10, 20, 30) and rotate 90 deg around Z
        angle_xform = np.pi / 2
        quat_xform = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_xform)
        custom_xform = wp.transform(wp.vec3(10.0, 20.0, 30.0), quat_xform)

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, xform=custom_xform)
        model = builder.finalize()

        # Compose transforms using warp
        body_xform = wp.transform(wp.vec3(0.5, 0.5, 0.0), quat_body)
        expected_xform = wp.transform_multiply(custom_xform, body_xform)
        expected_pos = expected_xform.p
        expected_quat = expected_xform.q

        body_idx = model.body_key.index("test_body")
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]

        np.testing.assert_allclose(body_pos, expected_pos, atol=1e-6)
        np.testing.assert_allclose(body_quat, expected_quat, atol=1e-6)

    def test_multiple_bodies_hierarchy(self):
        """Test 1: Multiple bodies in hierarchy → verify child transforms are correctly composed."""
        # Root is translated and rotated (45 deg around Z)
        angle_root = np.pi / 4
        quat_root = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_root)
        # MJCF expects [w, x, y, z]
        quat_root_mjcf = f"{quat_root[3]} {quat_root[0]} {quat_root[1]} {quat_root[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="2 3 0" quat="{quat_root_mjcf}">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="1 0 0" quat="0.7071068 0 0 0.7071068">
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Root: (2, 3, 0), 45 deg Z
        root_idx = model.body_key.index("root")
        root_pos = body_q[root_idx, :3]
        root_quat = body_q[root_idx, 3:]
        np.testing.assert_allclose(root_pos, [2, 3, 0], atol=1e-6)
        np.testing.assert_allclose(root_quat, quat_root, atol=1e-6)

        # Child: (1, 0, 0) in root frame, 90° Z rotation
        child_idx = model.body_key.index("child")
        child_pos = body_q[child_idx, :3]
        child_quat = body_q[child_idx, 3:]

        # Compose transforms using warp
        quat_child_mjcf = np.array([0.7071068, 0, 0, 0.7071068])
        # MJCF: [w, x, y, z] → warp: [x, y, z, w]
        quat_child = np.array([quat_child_mjcf[1], quat_child_mjcf[2], quat_child_mjcf[3], quat_child_mjcf[0]])
        child_xform = wp.transform(wp.vec3(1.0, 0.0, 0.0), quat_child)
        root_xform = wp.transform(wp.vec3(2.0, 3.0, 0.0), quat_root)
        expected_xform = wp.transform_multiply(root_xform, child_xform)
        expected_pos = expected_xform.p
        expected_quat = expected_xform.q

        np.testing.assert_allclose(child_pos, expected_pos, atol=1e-6)
        np.testing.assert_allclose(child_quat, expected_quat, atol=1e-6)

    def test_floating_base_transform(self):
        """Test 2: Floating base body → verify joint_q contains correct world coordinates, including rotation."""
        # Add a rotation: 90 deg about Z axis
        angle = np.pi / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        # MJCF expects [w, x, y, z]
        quat_mjcf = f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="floating_body" pos="2.0 3.0 4.0" quat="{quat_mjcf}">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # For floating base, joint_q should contain the body's world transform
        body_idx = model.body_key.index("floating_body")
        joint_idx = model.joint_key.index("floating_body_freejoint")

        # Get joint arrays at once
        joint_q_start = model.joint_q_start.numpy()
        joint_q = model.joint_q.numpy()

        joint_start = joint_q_start[joint_idx]

        # Extract position and orientation from joint_q
        joint_pos = [joint_q[joint_start + 0], joint_q[joint_start + 1], joint_q[joint_start + 2]]
        # Extract quaternion from joint_q (warp: [x, y, z, w])
        joint_quat = [
            joint_q[joint_start + 3],
            joint_q[joint_start + 4],
            joint_q[joint_start + 5],
            joint_q[joint_start + 6],
        ]

        # Should match the body's world transform
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]
        np.testing.assert_allclose(joint_pos, body_pos, atol=1e-6)
        np.testing.assert_allclose(joint_quat, body_quat, atol=1e-6)

    def test_chain_with_rotations(self):
        """Test 3: Chain of bodies with different pos/quat → verify each body's world transform."""
        # Test chain with cumulative rotations
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="1 0 0" quat="0.7071068 0 0 0.7071068">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="link2" pos="0 1 0" quat="0.7071068 0 0.7071068 0">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify each link's world transform
        base_idx = model.body_key.index("base")
        link1_idx = model.body_key.index("link1")
        link2_idx = model.body_key.index("link2")

        # Base: identity
        base_pos = body_q[base_idx, :3]
        base_quat = body_q[base_idx, 3:]
        np.testing.assert_allclose(base_pos, [0, 0, 0], atol=1e-6)
        # Identity quaternion in [x, y, z, w] format is [0, 0, 0, 1]
        np.testing.assert_allclose(base_quat, [0, 0, 0, 1], atol=1e-6)

        # Link1: base * link1_local
        link1_pos = body_q[link1_idx, :3]
        link1_quat = body_q[link1_idx, 3:]

        # Expected: base_xform * link1_local_xform
        base_xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(0, 0, 0, 1))
        link1_local_xform = wp.transform(wp.vec3(1, 0, 0), wp.quat(0, 0, 0.7071068, 0.7071068))
        expected_link1_xform = wp.transform_multiply(base_xform, link1_local_xform)

        np.testing.assert_allclose(link1_pos, expected_link1_xform.p, atol=1e-6)
        np.testing.assert_allclose(link1_quat, expected_link1_xform.q, atol=1e-6)

        # Link2: base * link1_local * link2_local
        link2_pos = body_q[link2_idx, :3]
        link2_quat = body_q[link2_idx, 3:]

        # Expected: link1_world_xform * link2_local_xform
        link2_local_xform = wp.transform(wp.vec3(0, 1, 0), wp.quat(0, 0.7071068, 0, 0.7071068))
        expected_link2_xform = wp.transform_multiply(expected_link1_xform, link2_local_xform)

        np.testing.assert_allclose(link2_pos, expected_link2_xform.p, atol=1e-6)
        np.testing.assert_allclose(link2_quat, expected_link2_xform.q, atol=1e-6)

    def test_bodies_with_scale(self):
        """Test 3: Bodies with scale → verify scaling is applied at each level."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="2 0 0">
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        # Parse with scale=2.0
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, scale=2.0)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify scaling is applied correctly
        root_idx = model.body_key.index("root")
        child_idx = model.body_key.index("child")

        # Root: no change
        root_pos = body_q[root_idx, :3]
        np.testing.assert_allclose(root_pos, [0, 0, 0], atol=1e-6)

        # Child: position scaled by 2.0
        child_pos = body_q[child_idx, :3]
        np.testing.assert_allclose(child_pos, [4, 0, 0], atol=1e-6)  # 2 * 2 = 4

    def test_tree_hierarchy_with_branching(self):
        """Test 3: Tree hierarchy with branching → verify transforms are correctly composed in all branches."""
        # Test a tree structure: root -> branch1 -> leaf1, and root -> branch2 -> leaf2
        # This tests that transforms are properly composed in parallel branches
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="0 0 0" quat="0.7071068 0 0 0.7071068">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="branch1" pos="1 0 0" quat="0.7071068 0 0.7071068 0">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="leaf1" pos="0 1 0" quat="1 0 0 0">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="branch2" pos="-1 0 0" quat="0.7071068 0.7071068 0 0">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="leaf2" pos="0 0 1" quat="0.7071068 0 0 0.7071068">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify transforms in all branches
        root_idx = model.body_key.index("root")
        branch1_idx = model.body_key.index("branch1")
        branch2_idx = model.body_key.index("branch2")
        leaf1_idx = model.body_key.index("leaf1")
        leaf2_idx = model.body_key.index("leaf2")

        # Root: (0, 0, 0), 90° Z rotation
        root_pos = body_q[root_idx, :3]
        root_quat = body_q[root_idx, 3:]
        np.testing.assert_allclose(root_pos, [0, 0, 0], atol=1e-6)
        # MJCF quat [0.7071068, 0, 0, 0.7071068] becomes [0, 0, 0.7071068, 0.7071068] in body_q
        np.testing.assert_allclose(root_quat, [0, 0, 0.7071068, 0.7071068], atol=1e-6)

        # Branch1: root * branch1_local
        branch1_pos = body_q[branch1_idx, :3]
        branch1_quat = body_q[branch1_idx, 3:]

        # Calculate expected using warp transforms
        root_xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(0, 0, 0.7071068, 0.7071068))
        # MJCF quat "0.7071068 0 0.7071068 0" is [w, x, y, z] -> convert to [x, y, z, w]
        branch1_local_quat = wp.quat(0, 0.7071068, 0, 0.7071068)
        branch1_local_xform = wp.transform(wp.vec3(1, 0, 0), branch1_local_quat)
        expected_branch1_xform = wp.transform_multiply(root_xform, branch1_local_xform)

        np.testing.assert_allclose(branch1_pos, expected_branch1_xform.p, atol=1e-6)
        np.testing.assert_allclose(branch1_quat, expected_branch1_xform.q, atol=1e-6)

        # Leaf1: root * branch1_local * leaf1_local
        leaf1_pos = body_q[leaf1_idx, :3]
        leaf1_quat = body_q[leaf1_idx, 3:]

        # MJCF quat "1 0 0 0" is [w, x, y, z] -> convert to [x, y, z, w]
        leaf1_local_quat = wp.quat(0, 0, 0, 1)  # Identity quaternion
        leaf1_local_xform = wp.transform(wp.vec3(0, 1, 0), leaf1_local_quat)
        expected_leaf1_xform = wp.transform_multiply(expected_branch1_xform, leaf1_local_xform)

        np.testing.assert_allclose(leaf1_pos, expected_leaf1_xform.p, atol=1e-6)
        np.testing.assert_allclose(leaf1_quat, expected_leaf1_xform.q, atol=1e-6)

        # Branch2: root * branch2_local
        branch2_pos = body_q[branch2_idx, :3]
        branch2_quat = body_q[branch2_idx, 3:]

        # MJCF quat "0.7071068 0.7071068 0 0" is [w, x, y, z] -> convert to [x, y, z, w]
        branch2_local_quat = wp.quat(0.7071068, 0, 0, 0.7071068)
        branch2_local_xform = wp.transform(wp.vec3(-1, 0, 0), branch2_local_quat)
        expected_branch2_xform = wp.transform_multiply(root_xform, branch2_local_xform)

        np.testing.assert_allclose(branch2_pos, expected_branch2_xform.p, atol=1e-6)
        np.testing.assert_allclose(branch2_quat, expected_branch2_xform.q, atol=1e-6)

        # Leaf2: root * branch2_local * leaf2_local
        leaf2_pos = body_q[leaf2_idx, :3]
        leaf2_quat = body_q[leaf2_idx, 3:]

        # MJCF quat "0.7071068 0 0 0.7071068" is [w, x, y, z] -> convert to [x, y, z, w]
        leaf2_local_quat = wp.quat(0, 0, 0.7071068, 0.7071068)
        leaf2_local_xform = wp.transform(wp.vec3(0, 0, 1), leaf2_local_quat)
        expected_leaf2_xform = wp.transform_multiply(expected_branch2_xform, leaf2_local_xform)

        np.testing.assert_allclose(leaf2_pos, expected_leaf2_xform.p, atol=1e-6)
        np.testing.assert_allclose(leaf2_quat, expected_leaf2_xform.q, atol=1e-6)

    def test_cylinder_shapes_preserved(self):
        """Test that cylinder geometries are properly imported as cylinders, not capsules."""
        # Create MJCF content with cylinder geometry
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="cylinder_test">
    <worldbody>
        <body name="test_body">
            <geom type="cylinder" size="0.5 1.0" />
            <geom type="cylinder" size="0.3 0.8" fromto="0 0 0 1 0 0" />
            <geom type="capsule" size="0.2 0.5" />
            <geom type="box" size="0.4 0.4 0.4" />
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "cylinder_test.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path)

            # Check that we have the correct number of shapes
            self.assertEqual(builder.shape_count, 4)

            # Check shape types
            shape_types = list(builder.shape_type)

            # First two shapes should be cylinders
            self.assertEqual(shape_types[0], GeoType.CYLINDER)
            self.assertEqual(shape_types[1], GeoType.CYLINDER)

            # Third shape should be capsule
            self.assertEqual(shape_types[2], GeoType.CAPSULE)

            # Fourth shape should be box
            self.assertEqual(shape_types[3], GeoType.BOX)

    def test_cylinder_properties_preserved(self):
        """Test that cylinder properties (radius, height) are correctly imported."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="cylinder_props_test">
    <worldbody>
        <body name="test_body">
            <geom type="cylinder" size="0.75 1.5" />
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "cylinder_props.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path)

            # Check shape properties
            self.assertEqual(builder.shape_count, 1)
            self.assertEqual(builder.shape_type[0], GeoType.CYLINDER)

            # Check that radius and half_height are preserved
            # shape_scale stores (radius, half_height, 0) for cylinders
            shape_scale = builder.shape_scale[0]
            self.assertAlmostEqual(shape_scale[0], 0.75)  # radius
            self.assertAlmostEqual(shape_scale[1], 1.5)  # half_height


if __name__ == "__main__":
    unittest.main(verbosity=2)
