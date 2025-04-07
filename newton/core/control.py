import warp as wp


class Control:
    """Time-varying control data for a :class:`Model`.

    Time-varying control data includes joint control inputs, muscle activations,
    and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`Model.control()` function.
    """

    def __init__(self):
        self.joint_act: wp.array | None = None
        """Array of joint control inputs with shape ``(joint_axis_count,)`` and type ``float``."""

        self.tri_activations: wp.array | None = None
        """Array of triangle element activations with shape ``(tri_count,)`` and type ``float``."""

        self.tet_activations: wp.array | None = None
        """Array of tetrahedral element activations with shape with shape ``(tet_count,) and type ``float``."""

        self.muscle_activations: wp.array | None = None
        """Array of muscle activations with shape ``(muscle_count,)`` and type ``float``."""

    def clear(self) -> None:
        """Reset the control inputs to zero."""

        if self.joint_act is not None:
            self.joint_act.zero_()
        if self.tri_activations is not None:
            self.tri_activations.zero_()
        if self.tet_activations is not None:
            self.tet_activations.zero_()
        if self.muscle_activations is not None:
            self.muscle_activations.zero_()

    def reset(self) -> None:
        """Reset the control inputs to zero."""

        wp.utils.warn(
            "Control.reset() is deprecated and will be removed\nin a future version. Use Control.clear() instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        self.clear()
