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

import itertools
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from fnmatch import fnmatch
from typing import Any

import numpy as np
import warp as wp

from ..sim import Contacts, Model
from ..solvers import SolverBase


class MatchKind(Enum):
    """Indicates the object type for a sensing object or a counterpart.

    - MATCH_ANY indicates a wildcard counterpart
    - SHAPE indicates a shape sensing object or counterpart
    - BODY indicates a body sensing object or counterpart
    """

    MATCH_ANY = 0
    SHAPE = 1
    BODY = 2


# TODO: merge with broadphase_common.binary_search
@wp.func
def bisect_shape_pairs(
    # inputs
    shape_pairs_sorted: wp.array(dtype=wp.vec2i),
    n_shape_pairs: wp.int32,
    value: wp.vec2i,
) -> wp.int32:
    lo = wp.int32(0)
    hi = n_shape_pairs
    while lo < hi:
        mid = (lo + hi) // 2
        pair_mid = shape_pairs_sorted[mid]
        if pair_mid[0] < value[0] or (pair_mid[0] == value[0] and pair_mid[1] < value[1]):
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def select_aggregate_net_force(
    # input
    num_contacts: wp.array(dtype=wp.int32),
    sp_sorted: wp.array(dtype=wp.vec2i),
    num_sp: int,
    sp_ep: wp.array(dtype=wp.vec2i),
    sp_ep_offset: wp.array(dtype=wp.int32),
    sp_ep_count: wp.array(dtype=wp.int32),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.float32),
    # output
    net_force: wp.array(dtype=wp.vec3),
):
    con_idx = wp.tid()
    if con_idx >= num_contacts[0]:
        return

    pair = contact_pair[con_idx]

    # Find the entity pairs
    smin, smax = wp.min(pair[0], pair[1]), wp.max(pair[0], pair[1])

    # add contribution for shape pair
    normalized_pair = wp.vec2i(smin, smax)
    sp_flip = normalized_pair[0] != pair[0]
    sp_ord = bisect_shape_pairs(sp_sorted, num_sp, normalized_pair)

    force = contact_force[con_idx] * contact_normal[con_idx]
    if sp_ord < num_sp and sp_sorted[sp_ord] == normalized_pair:
        # add the force to the pair's force accumulators
        offset = sp_ep_offset[sp_ord]
        for i in range(sp_ep_count[sp_ord]):
            ep = sp_ep[offset + i]
            force_acc, flip = ep[0], ep[1]
            wp.atomic_add(net_force, force_acc, wp.where(sp_flip != flip, -force, force))

    # add contribution for shape a and b
    for i in range(2):
        mono_sp = wp.vec2i(-1, pair[i])
        mono_ord = bisect_shape_pairs(sp_sorted, num_sp, mono_sp)

        # for shape vs all, only one accumulator is supported and flip is trivially true
        if mono_ord < num_sp and sp_sorted[mono_ord] == mono_sp:
            force_acc = sp_ep[sp_ep_offset[mono_ord]][0]
            wp.atomic_add(net_force, force_acc, wp.where(bool(i), -force, force))


class MatchAny:
    """Wildcard counterpart; matches any object."""


def populate_contacts(
    contacts: Contacts,
    solver: SolverBase,
):
    """
    Populate a Contacts object with the latest contact data from a solver.

    This function updates the given `contacts` object in-place using the contact information
    from the provided `solver`. It is typically called after a simulation step to refresh
    the contact data for use in sensors or analysis.

    Args:
        contacts (Contacts): The Contacts object to be populated or updated.
        solver (SolverBase): The solver instance containing the latest contact results.

    Returns:
        None
    """
    solver.update_contacts(contacts)


class ContactSensor:
    """Sensor for contact forces between bodies or shapes.

    The ContactSensor allows you to define a set of "sensing objects" (bodies or shapes) and optionally a set of
    "counterpart" objects (bodies or shapes) to sense contact forces against. The sensor can be configured to
    report the total contact force or per-counterpart readings.

    The ContactSensor produces a matrix of force readings, where each row corresponds to one sensing object and
    each column corresponds to one counterpart. Each entry of this matrix is the net contact force vector between
    the sensing object and the counterpart. If no counterparts are specified, the sensor will read the net contact
    force for each sensing object.

    If ``include_total`` is True, inserts a wildcard before all other counterparts, such that the first column of
    the force matrix will read the total contact force for each sensing object.

    If ``prune_noncolliding`` is True, the force matrix will be sparse, containing only readings for shape pairs that
    can collide. In this case, force matrix will have as many columns as the maximum number of active counterparts
    for any sensing object, and the ``reading_indices`` attribute can be used to recover the active counterparts
    for each sensing object.

    .. rubric:: Terms used

    - **Sensing Object**: The body or shape "carrying" a contact sensor.
    - **Counterpart**: The other body or shape involved in a contact interaction with a sensing object.
    - **Force Matrix**: The matrix organizing the force data by rows of sensing objects and columns of counterparts.
    - **Force Reading**: An individual force measurement within the matrix.

    Raises:
        ValueError: If the configuration of sensing/counterpart objects is invalid.
    """

    def __init__(
        self,
        model: Model,
        sensing_obj_bodies: str | list[str] | None = None,
        sensing_obj_shapes: str | list[str] | None = None,
        counterpart_bodies: str | list[str] | None = None,
        counterpart_shapes: str | list[str] | None = None,
        match_fn: Callable[[str, str], bool] | None = None,
        include_total: bool = True,
        prune_noncolliding: bool = False,
        verbose: bool | None = None,
    ):
        """Initialize a ContactSensor.

        Exactly one of ``sensing_obj_bodies`` or ``sensing_obj_shapes`` must be specified to define the sensing
        objects. At most one of ``counterpart_bodies`` or ``counterpart_shapes`` may be specified. If neither is
        specified, the sensor will read the net contact force for each sensing object.

        Args:
            sensing_obj_bodies: Pattern(s) to select which bodies are sensing objects.
            sensing_obj_shapes: Pattern(s) to select which shapes are sensing objects.
            counterpart_bodies: Pattern(s) to select which bodies are considered as counterparts.
            counterpart_shapes: Pattern(s) to select which shapes are considered as counterparts.
            match_fn: Function to match names to patterns. If None, uses ``fnmatch``.
            include_total: If True and counterparts are specified, add a reading for the total contact force for
                each sensing object. Does nothing when no counterparts are specified.
            prune_noncolliding: If True, omit force readings for shape pairs that never collide from the force
                matrix. Does nothing when no counterparts are specified.
            verbose: If True, print details. If None, uses ``wp.config.verbose``.
        """

        self.shape: tuple[int, int]
        """Shape of the force matrix (n_sensing_objs, n_counterparts) if ``prune_noncolliding`` is False, and
        (n_sensing_objs, max_active_counterparts) if it is True."""
        self.reading_indices: list[list[int]]
        """List of active counterpart indices per sensing object."""
        self.sensing_objs: list[tuple[int, MatchKind]]
        """Index and kind of each sensing object, length n_sensing_objs. Corresponds to the rows of the force matrix."""
        self.counterparts: list[tuple[int, MatchKind]]
        """Index and kind of each counterpart, length n_counterparts. Corresponds to the columns of the force matrix,
        unless ``prune_noncolliding`` is True."""

        self.net_force: wp.array2d(dtype=wp.vec3)
        """Net force matrix."""

        if (sensing_obj_bodies is None) == (sensing_obj_shapes is None):
            raise ValueError("Exactly one of `sensing_obj_bodies` and `sensing_obj_shapes` must be specified")

        if (counterpart_bodies is not None) and (counterpart_shapes is not None):
            raise ValueError("At most one of `counterpart_bodies` and `counterpart_shapes` may be specified.")

        self.device = model.device
        self.verbose = verbose if verbose is not None else wp.config.verbose

        if match_fn is None:
            match_fn = fnmatch

        if sensing_obj_bodies is not None:
            sensing_obj_bodies = self._match_elem_key(match_fn, model, model.body_key, sensing_obj_bodies)
            sensing_obj_shapes = []
        else:
            sensing_obj_bodies = []
            sensing_obj_shapes = self._match_elem_key(match_fn, model, model.shape_key, sensing_obj_shapes)

        if counterpart_bodies is not None:
            counterpart_bodies = self._match_elem_key(match_fn, model, model.body_key, counterpart_bodies)
            counterpart_shapes = []
            if include_total:
                counterpart_bodies = [MatchAny, *counterpart_bodies]
        elif counterpart_shapes is not None:
            counterpart_bodies = []
            counterpart_shapes = self._match_elem_key(match_fn, model, model.shape_key, counterpart_shapes)
            if include_total:
                counterpart_shapes = [MatchAny, *counterpart_shapes]
        else:
            counterpart_shapes = [MatchAny]
            counterpart_bodies = []

        sp_sorted, sp_reading, self.shape, self.reading_indices, self.sensing_objs, self.counterparts = (
            self._assemble_sensor_mappings(
                sensing_obj_bodies,
                sensing_obj_shapes,
                counterpart_bodies,
                counterpart_shapes,
                model.body_shapes,
                set(map(tuple, model.shape_contact_pairs.list())) if prune_noncolliding else None,
            )
        )

        # initialize warp arrays
        self._n_shape_pairs: int = len(sp_sorted)
        self._sp_sorted = wp.array(sp_sorted, dtype=wp.vec2i, device=self.device)
        self._sp_reading, self._sp_ep_offset, self._sp_ep_count = _lol_to_arrays(
            sp_reading, wp.vec2i, device=self.device
        )

        # net force (one vec3 per sensor-counterpart pair)
        self._net_force = wp.zeros(self.shape[0] * self.shape[1], dtype=wp.vec3, device=self.device)
        self.net_force = self._net_force.reshape(self.shape)

    def eval(self, contacts: Contacts):
        """Evaluate the contact sensor readings based on the provided contacts.

        Process the given Contacts object and updates the internal net force readings for each sensing_obj-counterpart
        pair.

        Args:
            contacts (Contacts): The contact data to evaluate.
        """
        self._eval_net_force(contacts)

    def get_total_force(self) -> wp.array2d(dtype=wp.vec3):
        """Get the total net force measured by the contact sensor.

        Returns:
            The net force array, shaped according to the sensor configuration.
        """
        return self.net_force

    @staticmethod
    def _assemble_sensor_mappings(
        sensing_obj_bodies: list[int],
        sensing_obj_shapes: list[int],
        counterpart_bodies: list[int | MatchAny],
        counterpart_shapes: list[int | MatchAny],
        body_shapes: dict[int, list[int]],
        shape_contact_pairs: set[tuple[int, int]] | None,
    ):
        # MatchAny, then bodies, then shapes
        def expand_bodies(bodies, shapes):
            has_matchany = MatchAny in bodies or MatchAny in shapes
            body_idx = [b for b in bodies if b is not MatchAny]
            shape_idx = [s for s in shapes if s is not MatchAny]
            body = [tuple(body_shapes[b]) for b in body_idx]
            shape = [(s,) for s in shape_idx]
            match_kind = (
                [MatchKind.MATCH_ANY] * has_matchany + [MatchKind.BODY] * len(body) + [MatchKind.SHAPE] * len(shape)
            )
            entities = [MatchAny] * has_matchany + body + shape
            indices = [MatchAny] * has_matchany + body_idx + shape_idx
            return list(zip(indices, match_kind, strict=False)), entities

        def get_colliding_sps(a, b) -> dict[tuple[int, int], bool]:
            all_pairs_flip = {
                (min(pair), max(pair)): min(pair) == pair[1] for pair in itertools.product(a, b) if pair[0] != pair[1]
            }
            if shape_contact_pairs is None:
                return all_pairs_flip
            return {pair: all_pairs_flip[pair] for pair in shape_contact_pairs.intersection(all_pairs_flip)}

        sensing_obj_kinds, sensing_objs = expand_bodies(sensing_obj_bodies, sensing_obj_shapes)
        counterpart_kinds, counterparts = expand_bodies(counterpart_bodies, counterpart_shapes)
        counterpart_indices = []
        sp_to_reading = defaultdict(list)

        # build list of counterpart indices for each sensing_obj
        # build list of shape pairs for each reading of each sensing_obj
        # build the mapping from shape pair to tuples of reading index and flip indicator
        # the mapping is ordered lexicographically by sorted shape pair
        for sensing_obj_idx, sensing_obj in enumerate(sensing_objs):
            if sensing_obj is MatchAny:
                raise ValueError("Sensing object cannot be MatchAny")
            sens_counterparts: list[tuple[int, MatchKind], ...] = []
            reading_idx = 0
            for counterpart_idx, counterpart in enumerate(counterparts):
                if counterpart is MatchAny:
                    sp_flips = dict.fromkeys(itertools.product((-1,), sensing_obj), True)
                elif not (sp_flips := get_colliding_sps(sensing_obj, counterpart)):
                    continue

                for sp, flip in sp_flips.items():
                    sp_to_reading[sp].append((sensing_obj_idx, reading_idx, flip))
                sens_counterparts.append(counterpart_idx)
                reading_idx += 1
            counterpart_indices.append(sens_counterparts)

        # maximum number of readings for any sensing object
        n_readings = max(map(len, counterpart_indices))

        sp_sorted = sorted(sp_to_reading)
        sp_reading = []
        for sp in sp_sorted:
            sp_reading.append(
                [
                    (sensing_obj_idx * n_readings + reading_idx, flip)
                    for sensing_obj_idx, reading_idx, flip in sp_to_reading[sp]
                ]
            )

        shape = len(sensing_objs), n_readings
        return sp_sorted, sp_reading, shape, counterpart_indices, sensing_obj_kinds, counterpart_kinds

    def _eval_net_force(self, contact: Contacts):
        self._net_force.zero_()
        wp.launch(
            select_aggregate_net_force,
            dim=contact.rigid_contact_max,
            inputs=[
                contact.rigid_contact_count,
                self._sp_sorted,
                self._n_shape_pairs,
                self._sp_reading,
                self._sp_ep_offset,
                self._sp_ep_count,
                contact.pair,
                contact.normal,
                contact.force,
            ],
            outputs=[self._net_force],
            device=contact.device,
        )

    @classmethod
    def _match_elem_key(
        cls,
        match_fn: Callable[[str, str], bool],
        model: Model,
        elem_key: dict[str, Any],
        pattern: str | list[str],
    ) -> list[int]:
        """Find the indices of elements matching the pattern."""
        matches = []

        if isinstance(pattern, list):
            for single_pattern in pattern:
                matches.extend(cls._match_elem_key(match_fn, model, elem_key, single_pattern))
            return matches

        for idx, elem in enumerate(elem_key):
            if match_fn(elem, pattern):
                matches.append(idx)

        return matches


def _lol_to_arrays(list_of_lists: list[list], dtype, **kwargs) -> tuple[wp.array, wp.array, wp.array]:
    """Convert a list of lists to three warp arrays containing the values, offsets and counts.
    Does nothing and returns None, None, None if the list is empty.
    """
    if not list_of_lists:
        return None, None, None
    value_list = [val for l in list_of_lists for val in l]
    count_list = [len(l) for l in list_of_lists]

    values = wp.array(value_list, dtype=dtype, **kwargs)
    offset = wp.array(np.cumsum([0, *count_list[:-1]]), dtype=wp.int32, **kwargs)
    count = wp.array(count_list, dtype=wp.int32, **kwargs)
    return values, offset, count
