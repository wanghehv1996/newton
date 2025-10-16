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
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import find_nan_members


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory() -> str:
    return os.path.join(get_source_directory(), "assets")


def get_asset(filename: str) -> str:
    return os.path.join(get_asset_directory(), filename)


def test_body_state(
    model: newton.Model,
    state: newton.State,
    test_name: str,
    test_fn: wp.Function | Callable[[wp.transform, wp.spatial_vectorf], bool],
    indices: list[int] | None = None,
    show_body_q: bool = False,
    show_body_qd: bool = False,
):
    """
    Test the position and velocity coordinates of the given bodies by applying the given test function to each body.
    The function will raise a ``ValueError`` if the test fails for any of the given bodies.

    Args:
        model: The model to test.
        state: The state to test.
        test_name: The name of the test.
        test_fn: The test function to evaluate. Maps from the body pose and twist to a boolean.
        indices: The indices of the bodies to test. If None, all bodies will be tested.
        show_body_q: Whether to print the body pose in the error message.
        show_body_qd: Whether to print the body twist in the error message.
    """

    # construct a Warp kernel to evaluate the test function for the given body indices
    if isinstance(test_fn, wp.Function):
        warp_test_fn = test_fn
    else:
        warp_test_fn, _ = wp.utils.create_warp_function(test_fn)
    if indices is None:
        indices = np.arange(model.body_count, dtype=np.int32).tolist()

    @wp.kernel
    def test_fn_kernel(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        indices: wp.array(dtype=int),
        # output
        failures: wp.array(dtype=bool),
    ):
        env_id = wp.tid()
        index = indices[env_id]
        failures[env_id] = not warp_test_fn(body_q[index], body_qd[index])

    body_q = state.body_q
    body_qd = state.body_qd
    if body_q is None or body_qd is None:
        raise ValueError("Body state is not available")
    with wp.ScopedDevice(body_q.device):
        failures = wp.zeros(len(indices), dtype=bool)
        indices_array = wp.array(indices, dtype=int)
        wp.launch(
            test_fn_kernel,
            dim=len(indices),
            inputs=[body_q, body_qd, indices_array],
            outputs=[failures],
        )
        failures_np = failures.numpy()
        if np.any(failures_np):
            body_key = np.array(model.body_key)[indices]
            body_q = body_q.numpy()[indices]
            body_qd = body_qd.numpy()[indices]
            failed_indices = np.where(failures_np)[0]
            failed_details = []
            for index in failed_indices:
                detail = body_key[index]
                extras = []
                if show_body_q:
                    extras.append(f"q={body_q[index]}")
                if show_body_qd:
                    extras.append(f"qd={body_qd[index]}")
                if len(extras) > 0:
                    failed_details.append(f"{detail} ({', '.join(extras)})")
                else:
                    failed_details.append(detail)
            raise ValueError(f'Test "{test_name}" failed for the following bodies: [{", ".join(failed_details)}]')


def test_particle_state(
    state: newton.State,
    test_name: str,
    test_fn: wp.Function | Callable[[wp.vec3, wp.vec3], bool],
    indices: list[int] | None = None,
):
    """
    Test the position and velocity coordinates of the given particles by applying the given test function to each particle.
    The function will raise a ``ValueError`` if the test fails for any of the given particles.

    Args:
        state: The state to test.
        test_name: The name of the test.
        test_fn: The test function to evaluate. Maps from the particle position and velocity to a boolean.
        indices: The indices of the particles to test. If None, all particles will be tested.
    """

    # construct a Warp kernel to evaluate the test function for the given body indices
    if isinstance(test_fn, wp.Function):
        warp_test_fn = test_fn
    else:
        warp_test_fn, _ = wp.utils.create_warp_function(test_fn)
    if indices is None:
        indices = np.arange(state.particle_count, dtype=np.int32).tolist()

    @wp.kernel
    def test_fn_kernel(
        particle_q: wp.array(dtype=wp.vec3),
        particle_qd: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=int),
        # output
        failures: wp.array(dtype=bool),
    ):
        env_id = wp.tid()
        index = indices[env_id]
        failures[env_id] = not warp_test_fn(particle_q[index], particle_qd[index])

    particle_q = state.particle_q
    particle_qd = state.particle_qd
    if particle_q is None or particle_qd is None:
        raise ValueError("Particle state is not available")
    with wp.ScopedDevice(particle_q.device):
        failures = wp.zeros(len(indices), dtype=bool)
        indices_array = wp.array(indices, dtype=int)
        wp.launch(
            test_fn_kernel,
            dim=len(indices),
            inputs=[particle_q, particle_qd, indices_array],
            outputs=[failures],
        )
        failures_np = failures.numpy()
        if np.any(failures_np):
            failed_particles = np.where(failures_np)[0]
            raise ValueError(f'Test "{test_name}" failed for {len(failed_particles)} out of {len(indices)} particles')


def run(example, args):
    if hasattr(example, "gui") and hasattr(example.viewer, "register_ui_callback"):
        example.viewer.register_ui_callback(lambda ui: example.gui(ui), position="side")

    while example.viewer.is_running():
        if not example.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                example.step()

        with wp.ScopedTimer("render", active=False):
            example.render()

    if args is not None and args.test:
        if not hasattr(example, "test"):
            raise NotImplementedError("Example does not have a test method")
        example.test()

    example.viewer.close()

    if args is not None and args.test:
        # generic tests for finiteness of Newton objects
        if hasattr(example, "state_0"):
            nan_members = find_nan_members(example.state_0)
            if nan_members:
                raise ValueError(f"NaN members found in state_0: {nan_members}")
        if hasattr(example, "state_1"):
            nan_members = find_nan_members(example.state_1)
            if nan_members:
                raise ValueError(f"NaN members found in state_1: {nan_members}")
        if hasattr(example, "model"):
            nan_members = find_nan_members(example.model)
            if nan_members:
                raise ValueError(f"NaN members found in model: {nan_members}")
        if hasattr(example, "control"):
            nan_members = find_nan_members(example.control)
            if nan_members:
                raise ValueError(f"NaN members found in control: {nan_members}")
        if hasattr(example, "contacts"):
            nan_members = find_nan_members(example.contacts)
            if nan_members:
                raise ValueError(f"NaN members found in contacts: {nan_members}")


def compute_env_offsets(
    num_envs: int, env_offset: tuple[float, float, float] = (5.0, 5.0, 0.0), up_axis: newton.AxisType = newton.Axis.Z
):
    # raise deprecation warning
    import warnings  # noqa: PLC0415

    warnings.warn(
        (
            "compute_env_offsets is deprecated and will be removed in a future version. "
            "Use the builder.replicate() function instead."
        ),
        stacklevel=2,
    )

    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
        if num_dim == 1:
            for i in range(num_envs):
                env_offsets.append(i * env_offset)
        elif num_dim == 2:
            for i in range(num_envs):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
                offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
                env_offsets.append(offset)
        elif num_dim == 3:
            for i in range(num_envs):
                d0 = i // (side_length * side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * env_offset[0]
                offset[1] = d1 * env_offset[1]
                offset[2] = d2 * env_offset[2]
                env_offsets.append(offset)
        env_offsets = np.array(env_offsets)
    else:
        env_offsets = np.zeros((num_envs, 3))
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    # ensure the envs are not shifted below the ground plane
    correction[newton.Axis.from_any(up_axis)] = 0.0
    env_offsets -= correction
    return env_offsets


def create_parser():
    """Create a base argument parser with common parameters for Newton examples.

    Individual examples can use this as a parent parser and add their own
    specific arguments.

    Returns:
        argparse.ArgumentParser: Base parser with common arguments
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "rerun", "null"],
        help="Viewer to use (gl, usd, rerun, or null).",
    )
    parser.add_argument(
        "--output-path", type=str, default="output.usd", help="Path to the output USD file (required for usd viewer)."
    )
    parser.add_argument("--num-frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to initialize the viewer headless (for OpenGL viewer only).",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the example in test mode.",
    )

    return parser


def init(parser=None):
    """Initialize Newton example components from parsed arguments.

    Args:
        parser: Parsed arguments from argparse (should include arguments from
              create_parser())

    Returns:
        tuple: (viewer, args) where viewer is configured based on args.viewer

    Raises:
        ValueError: If invalid viewer type or missing required arguments
    """
    import warp as wp  # noqa: PLC0415

    import newton.viewer  # noqa: PLC0415

    # parse args
    if parser is None:
        parser = create_parser()
        args = parser.parse_known_args()[0]
    else:
        # When parser is provided, use parse_args() to properly handle --help
        args = parser.parse_args()

    # Set device if specified
    if args.device:
        wp.set_device(args.device)

    # Create viewer based on type
    if args.viewer == "gl":
        viewer = newton.viewer.ViewerGL(headless=args.headless)
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun()
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    return viewer, args


def main():
    """Main entry point for running examples via 'python -m newton.examples <example_name>'."""
    import runpy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # Map short names to full module paths
    example_map = {}
    modules = ["basic", "cloth", "diffsim", "ik", "mpm", "robot", "selection", "sensors"]
    for module in sorted(modules):
        for example in sorted(os.listdir(os.path.join(get_source_directory(), module))):
            if example.endswith(".py"):
                example_name = example[8:-3]  # Remove "example_" prefix and ".py" file ext
                example_map[example_name] = f"newton.examples.{module}.{example[:-3]}"

    if len(sys.argv) < 2:
        print("Usage: python -m newton.examples <example_name>")
        print("\nAvailable examples:")
        for name in example_map.keys():
            print(f"  {name}")
        sys.exit(1)

    example_name = sys.argv[1]

    if example_name not in example_map:
        print(f"Error: Unknown example '{example_name}'")
        print("\nAvailable examples:")
        for name in example_map.keys():
            print(f"  {name}")
        sys.exit(1)

    # Set up sys.argv for the target script
    target_module = example_map[example_name]
    # Keep the module name as argv[0] and pass remaining args
    sys.argv = [target_module, *sys.argv[2:]]

    # Run the target example module
    runpy.run_module(target_module, run_name="__main__")


if __name__ == "__main__":
    main()


__all__ = ["create_parser", "init", "run", "test_body_state", "test_particle_state"]
