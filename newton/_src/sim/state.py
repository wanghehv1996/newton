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

from __future__ import annotations

import warp as wp


class State:
    """
    Represents the time-varying state of a :class:`Model` in a simulation.

    The State object holds all dynamic quantities that change over time during simulation,
    such as particle and rigid body positions, velocities, and forces, as well as joint coordinates.

    State objects are typically created via :meth:`newton.Model.state()` and are used to
    store and update the simulation's current configuration and derived data.
    """

    def __init__(self) -> None:
        self.particle_q: wp.array | None = None
        """3D positions of particles, shape (particle_count,), dtype :class:`vec3`."""

        self.particle_qd: wp.array | None = None
        """3D velocities of particles, shape (particle_count,), dtype :class:`vec3`."""

        self.particle_f: wp.array | None = None
        """3D forces on particles, shape (particle_count,), dtype :class:`vec3`."""

        self.body_q: wp.array | None = None
        """Rigid body transforms (7-DOF), shape (body_count,), dtype :class:`transform`."""

        self.body_qd: wp.array | None = None
        """Rigid body velocities (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear velocity; last three: angular velocity."""

        self.body_f: wp.array | None = None
        """Rigid body forces (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear force; last three: torque.

        Note:
            :attr:`body_f` represents external wrenches in world frame, measured at the body's center of mass
            for all solvers except :class:`~newton.solvers.SolverFeatherstone`, which expects wrenches at the world origin.
        """

        self.joint_q: wp.array | None = None
        """Generalized joint position coordinates, shape (joint_coord_count,), dtype float."""

        self.joint_qd: wp.array | None = None
        """Generalized joint velocity coordinates, shape (joint_dof_count,), dtype float."""

    def clear_forces(self) -> None:
        """
        Clear all force arrays (for particles and bodies) in the state object.

        Sets all entries of :attr:`particle_f` and :attr:`body_f` to zero, if present.
        """
        with wp.ScopedTimer("clear_forces", False):
            if self.particle_count:
                self.particle_f.zero_()

            if self.body_count:
                self.body_f.zero_()

    @property
    def requires_grad(self) -> bool:
        """Indicates whether the state arrays have gradient computation enabled."""
        if self.particle_q:
            return self.particle_q.requires_grad
        if self.body_q:
            return self.body_q.requires_grad
        return False

    @property
    def body_count(self) -> int:
        """The number of bodies represented in the state."""
        if self.body_q is None:
            return 0
        return len(self.body_q)

    @property
    def particle_count(self) -> int:
        """The number of particles represented in the state."""
        if self.particle_q is None:
            return 0
        return len(self.particle_q)

    @property
    def joint_coord_count(self) -> int:
        """The number of generalized joint position coordinates represented in the state."""
        if self.joint_q is None:
            return 0
        return len(self.joint_q)

    @property
    def joint_dof_count(self) -> int:
        """The number of generalized joint velocity coordinates represented in the state."""
        if self.joint_qd is None:
            return 0
        return len(self.joint_qd)
