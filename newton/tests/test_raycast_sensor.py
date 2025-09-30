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

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import RaycastSensor
from newton.tests.unittest_utils import add_function_test, get_test_devices

EXPORT_IMAGES = False


def save_depth_image_as_grayscale(depth_image: np.ndarray, filename: str):
    """Save a depth image as a grayscale image.

    Args:
        depth_image: 2D numpy array with depth values (-1.0 for no hit, positive for distances)
        filename: Name of the file (without extension)
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return  # Skip if PIL not available

    # Handle the depth image: -1.0 means no hit, positive values are distances
    img_data = depth_image.copy().astype(np.float32)

    # Replace -1.0 (no hit) with 0 (black)
    img_data[img_data < 0] = 0

    # Normalize positive values to 0-255 range
    pos_mask = img_data > 0
    if np.any(pos_mask):
        pos_vals = img_data[pos_mask]
        min_depth = pos_vals.min()
        max_depth = pos_vals.max()
        denom = max(max_depth - min_depth, 1e-6)
        # Invert: closer objects = brighter, farther = darker
        # Scale to 50-255 range (so background/no-hit stays at 0)
        img_data[pos_mask] = 255 - ((pos_vals - min_depth) / denom) * 205

    # Convert to uint8 and save
    img_data = np.clip(img_data, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_data)

    filepath = f"{filename}.png"
    image.save(filepath)


def create_cubemap_scene(device="cpu"):
    """Create a scene with 6 different objects positioned around origin for cube map views."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    # Position objects at distance ~6 units from origin in different directions
    # This ensures each cube map face sees a different object

    # Capsule: positioned in +X direction
    capsule_body = builder.add_body(xform=wp.transform(wp.vec3(6.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_capsule(body=capsule_body, radius=0.8, half_height=1.5)

    # Sphere: positioned in -X direction
    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(-6.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(body=sphere_body, radius=1.2)

    # Cone: positioned in +Y direction
    cone_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 6.0, 0.0), wp.quat_identity()))
    builder.add_shape_cone(body=cone_body, radius=1.1, half_height=1.3)

    # Cylinder: positioned in -Y direction
    cylinder_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, -6.0, 0.0), wp.quat_identity()))
    builder.add_shape_cylinder(body=cylinder_body, radius=0.9, half_height=1.2)

    # Cube: positioned in +Z direction
    cube_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 6.0), wp.quat_identity()))
    builder.add_shape_box(body=cube_body, hx=1.0, hy=1.0, hz=1.0)

    # Tetrahedron mesh: positioned in -Z direction
    tetrahedron_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -6.0), wp.quat_identity()))

    # Create tetrahedron mesh vertices and faces
    # Regular tetrahedron with vertices at distance ~1.5 from center
    s = 1.5  # Scale factor
    vertices = np.array(
        [
            [s, s, s],  # vertex 0
            [s, -s, -s],  # vertex 1
            [-s, s, -s],  # vertex 2
            [-s, -s, s],  # vertex 3
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            [0, 1, 2],  # face 0
            [0, 3, 1],  # face 1
            [0, 2, 3],  # face 2
            [1, 3, 2],  # face 3
        ],
        dtype=np.int32,
    )

    # Create newton Mesh object and add to builder
    tetrahedron_mesh = newton.Mesh(vertices, faces.flatten())
    builder.add_shape_mesh(body=tetrahedron_body, mesh=tetrahedron_mesh, scale=(1.0, 1.0, 1.0))

    # Build the model
    with wp.ScopedDevice(device):
        model = builder.finalize()
    return model


def test_raycast_sensor_cubemap(test: unittest.TestCase, device, export_images: bool = False):
    """Test raycast sensor by creating cube map views from origin."""

    # Create scene with 6 different objects (one for each cube map face)
    model = create_cubemap_scene(device)
    state = model.state()

    # Update body transforms (important for raycast operations)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Define 6 cube map camera directions
    cubemap_views = [
        ("positive_x", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # Looking +X (capsule)
        ("negative_x", (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # Looking -X (sphere)
        ("positive_y", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # Looking +Y (cone)
        ("negative_y", (0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # Looking -Y (cylinder)
        ("positive_z", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # Looking +Z (cube)
        ("negative_z", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),  # Looking -Z (tetrahedron)
    ]

    # Create raycast sensor (we'll update camera parameters for each view)
    sensor = RaycastSensor(
        model=model,
        camera_position=(0.0, 0.0, 0.0),  # At origin
        camera_direction=(1.0, 0.0, 0.0),  # Initial direction (will be updated)
        camera_up=(0.0, 0.0, 1.0),  # Initial up (will be updated)
        fov_radians=math.pi / 2,  # 90 degrees - typical for cube map faces
        width=256,
        height=256,
        max_distance=50.0,
    )

    # Render each cube map face
    for view_name, position, direction, up in cubemap_views:
        # Update camera pose for this view
        sensor.update_camera_pose(position=position, direction=direction, up=up)

        # Evaluate the sensor
        sensor.eval(state)

        # Get depth image
        depth_image = sensor.get_depth_image_numpy()

        # Count hits for this view
        hits_in_view = np.sum(depth_image > 0)

        # Verify each face has at least one hit
        test.assertGreater(hits_in_view, 0, f"Face {view_name} should detect at least one object hit")

        # Save depth image (if enabled)
        if EXPORT_IMAGES:
            save_depth_image_as_grayscale(depth_image, f"cubemap_{view_name}")


class TestRaycastSensor(unittest.TestCase):
    pass


# Register test for all available devices
devices = get_test_devices()
add_function_test(TestRaycastSensor, "test_raycast_sensor_cubemap", test_raycast_sensor_cubemap, devices=devices)


if __name__ == "__main__":
    unittest.main()
