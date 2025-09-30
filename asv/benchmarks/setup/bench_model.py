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

import gc

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if

from newton.examples.example_mujoco import Example


class KpiInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [4096, 8192])
    param_names = ["robot", "num_envs"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1
    timeout = 3600

    def setup(self, robot, num_envs):
        wp.init()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, robot, num_envs):
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        _model = builder.finalize()
        wp.synchronize_device()


class KpiInitializeSolver:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [4096, 8192])
    param_names = ["robot", "num_envs"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1
    timeout = 3600

    def setup(self, robot, num_envs):
        if robot == "h1":
            # use more samples for H1 to reduce variance
            self.repeat = 10

        wp.init()
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_solver(self, robot, num_envs):
        self._solver = Example.create_solver(self._model, robot, use_mujoco_cpu=False)
        wp.synchronize_device()

    def teardown(self, robot, num_envs):
        del self._solver
        del self._model


class FastInitializeModel:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [128, 256])
    param_names = ["robot", "num_envs"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup_cache(self):
        # Load a small model to cache the kernels
        builder = Example.create_model_builder("cartpole", 1, randomize=False, seed=123)
        model = builder.finalize(device="cpu")
        del model

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, robot, num_envs):
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        _model = builder.finalize()
        wp.synchronize_device()

    def peakmem_initialize_model_cpu(self, robot, num_envs):
        gc.collect()

        with wp.ScopedDevice("cpu"):
            builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

            # finalize model
            model = builder.finalize()

        del model


class FastInitializeSolver:
    params = (["humanoid", "g1", "h1", "cartpole", "ant", "quadruped"], [128, 256])
    param_names = ["robot", "num_envs"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, num_envs):
        if robot == "h1":
            # use more samples for H1 to reduce variance
            self.repeat = 10

        wp.init()
        builder = Example.create_model_builder(robot, num_envs, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_solver(self, robot, num_envs):
        self._solver = Example.create_solver(self._model, robot, use_mujoco_cpu=False)
        wp.synchronize_device()

    def teardown(self, robot, num_envs):
        del self._solver
        del self._model


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiInitializeModel": KpiInitializeModel,
        "FastInitializeModel": FastInitializeModel,
        "KpiInitializeSolver": KpiInitializeSolver,
        "FastInitializeSolver": FastInitializeSolver,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
