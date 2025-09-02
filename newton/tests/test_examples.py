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

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. There are no
correctness checks.

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import warp as wp

import newton.tests.unittest_utils
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)

wp.init()


def _build_command_line_options(test_options: dict[str, Any]) -> list:
    """Helper function to build command-line options from the test options dictionary."""
    additional_options = []

    for key, value in test_options.items():
        if isinstance(value, bool):
            # Default behavior expecting argparse.BooleanOptionalAction support
            additional_options.append(f"--{'no-' if not value else ''}{key.replace('_', '-')}")
        if isinstance(value, list):
            additional_options.extend([f"--{key.replace('_', '-')}"] + [str(v) for v in value])
        else:
            # Just add --key value
            additional_options.extend(["--" + key.replace("_", "-"), str(value)])

    return additional_options


def _merge_options(base_options: dict[str, Any], device_options: dict[str, Any]) -> dict[str, Any]:
    """Helper function to merge base test options with device-specific test options."""
    merged_options = base_options.copy()

    #  Update options with device-specific dictionary, overwriting existing keys with the more-specific values
    merged_options.update(device_options)
    return merged_options


def add_example_test(
    cls: type,
    name: str,
    devices: list | None = None,
    test_options: dict[str, Any] | None = None,
    test_options_cpu: dict[str, Any] | None = None,
    test_options_cuda: dict[str, Any] | None = None,
    use_viewer: bool = False,
    test_suffix: str | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase."""

    if test_options is None:
        test_options = {}
    if test_options_cpu is None:
        test_options_cpu = {}
    if test_options_cuda is None:
        test_options_cuda = {}

    def run(test, device):
        if wp.get_device(device).is_cuda:
            options = _merge_options(test_options, test_options_cuda)
        else:
            options = _merge_options(test_options, test_options_cpu)

        # Mark the test as skipped if Torch is not installed but required
        torch_required = options.pop("torch_required", False)
        if torch_required:
            try:
                import torch  # noqa: PLC0415

                if wp.get_device(device).is_cuda and not torch.cuda.is_available():
                    # Ensure torch has CUDA support
                    test.skipTest("Torch not compiled with CUDA support")

            except Exception as e:
                test.skipTest(f"{e}")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Find the current Warp cache
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = warp_cache_path

        if newton.tests.unittest_utils.coverage_enabled:
            # Generate a random coverage data file name - file is deleted along with containing directory
            with tempfile.NamedTemporaryFile(
                dir=newton.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = ["coverage", "run", f"--data-file={coverage_file.name}"]

            if newton.tests.unittest_utils.coverage_branch:
                command.append("--branch")

        else:
            command = [sys.executable]

        # Append Warp commands
        command.extend(["-m", f"newton.examples.{name}", "--device", str(device)])

        if not use_viewer:
            stage_path = (
                options.pop(
                    "stage_path",
                    os.path.join(os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"),
                )
                if USD_AVAILABLE
                else "None"
            )

            if stage_path:
                command.extend(["--stage-path", stage_path])
                try:
                    os.remove(stage_path)
                except OSError:
                    pass
        else:
            # new-style example, setup viewer type and output path
            if USD_AVAILABLE:
                stage_path = os.path.join(
                    os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"
                )
                command.extend(["--viewer", "usd", "--output-path", stage_path])
            else:
                stage_path = "None"
                command.extend(["--viewer", "null"])

        command.extend(_build_command_line_options(options))

        # Set the test timeout in seconds
        test_timeout = options.pop("test_timeout", 600)

        # Can set active=True when tuning the test parameters
        with wp.ScopedTimer(f"{name}_{sanitize_identifier(device)}", active=False):
            # Run the script as a subprocess
            result = subprocess.run(
                command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
            )

        # print any error messages (e.g.: module not found)
        if result.stderr != "":
            print(result.stderr)

        # Check the return code (0 is standard for success)
        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # If the test succeeded, try to clean up the output by default
        if stage_path and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(unittest.TestCase):
    pass


add_example_test(TestBasicExamples, name="basic.example_basic_pendulum", devices=test_devices, use_viewer=True)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options_cpu={"num_envs": 16},
    test_options_cuda={"num_envs": 64},
    use_viewer=True,
)

add_example_test(TestBasicExamples, name="basic.example_basic_viewer", devices=test_devices, use_viewer=True)

add_example_test(TestBasicExamples, name="basic.example_basic_joints", devices=test_devices, use_viewer=True)

add_example_test(TestBasicExamples, name="basic.example_basic_shapes", devices=test_devices, use_viewer=True)


class TestClothExamples(unittest.TestCase):
    pass


add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_bending",
    devices=test_devices,
    test_options={"num_frames": 100},
    test_options_cpu={"num_frames": 100},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={},
    test_options_cpu={"width": 32, "height": 16, "num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_style3d",
    devices=test_devices,
    test_options={},
    test_options_cuda={"num_frames": 32},
    test_options_cpu={"num_frames": 2},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_franka",
    devices=test_devices,
    test_options={"num_frames": 50},
    test_options_cpu={"num_frames": 2},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_twist",
    devices=test_devices,
    test_options={"num_frames": 100},
    test_options_cpu={"num_frames": 20},
    use_viewer=True,
)


class TestRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotExamples,
    name="robot.example_robot_cartpole",
    devices=test_devices,
    test_options={"usd_required": True, "num_frames": 100},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_c_walk",
    devices=test_devices,
    test_options={"usd_required": True, "num_frames": 500, "torch_required": True},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=test_devices,
    test_options={"usd_required": True, "num_frames": 500},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=test_devices,
    test_options={"usd_required": True, "num_frames": 500},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=test_devices,
    test_options={"usd_required": True, "num_frames": 500},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_humanoid",
    devices=test_devices,
    test_options={"num_frames": 500},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)


class TestRobotPolicyExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num_frames": 500, "torch_required": True, "robot": "g1_29dof"},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
    test_suffix="G1_29dof",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num_frames": 500, "torch_required": True, "robot": "g1_23dof"},
    use_viewer=True,
    test_suffix="G1_23dof",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num_frames": 500, "torch_required": True, "robot": "g1_23dof", "physx": True},
    use_viewer=True,
    test_suffix="G1_23dof_Physx",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num_frames": 500, "torch_required": True, "robot": "anymal"},
    use_viewer=True,
    test_suffix="Anymal",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num_frames": 500, "torch_required": True, "robot": "anymal", "physx": True},
    use_viewer=True,
    test_suffix="Anymal_Physx",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num_frames": 500, "robot": "go2"},
    use_viewer=True,
    test_suffix="Go2",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num_frames": 500, "robot": "go2", "physx": True},
    use_viewer=True,
    test_suffix="Go2_Physx",
)


class TestAdvancedRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num_frames": 100, "torch_required": True},
    use_viewer=True,
)


class TestIKExamples(unittest.TestCase):
    pass


add_example_test(TestIKExamples, name="ik.example_ik_franka", devices=test_devices, use_viewer=True)

add_example_test(TestIKExamples, name="ik.example_ik_h1", devices=test_devices, use_viewer=True)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_benchmark",
    devices=test_devices,
    test_options_cpu={"batch_sizes": [1, 10]},
    use_viewer=True,
)


class TestSelectionAPIExamples(unittest.TestCase):
    pass


add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_articulations",
    devices=test_devices,
    test_options={"num_frames": 100},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_cartpole",
    devices=test_devices,
    test_options={"num_frames": 100},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=test_devices,
    test_options={"num_frames": 100},
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)


class TestDiffSimExamples(unittest.TestCase):
    pass


add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num_frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num_frames": 2 * 36},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num_frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num_frames": 2 * 120},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num_frames": 180},  # sim_steps
    test_options_cpu={"num_frames": 10},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num_frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num_frames": 2 * 30},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num_frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num_frames": 2 * 60},
    use_viewer=True,
)


class TestOtherExamples(unittest.TestCase):
    pass


add_example_test(
    TestOtherExamples,
    name="mpm.example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"viewer": "null", "num_frames": 100},
    use_viewer=True,
)

add_example_test(
    TestOtherExamples,
    name="example_rigid_force",
    devices=test_devices,
    test_options={"headless": True},
)

add_example_test(
    TestOtherExamples,
    name="example_contact_sensor",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 100},
    test_options_cpu={"num_frames": 10},
)


if __name__ == "__main__":
    # force rebuild of all kernels
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2)
