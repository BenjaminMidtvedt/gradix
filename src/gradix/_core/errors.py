"""Errors raised by gradix; each one says what went wrong and how to fix it."""

from __future__ import annotations

__all__ = [
    "BindingError",
    "CapacityError",
    "EnvelopeError",
    "GradientPathError",
    "GradixError",
    "GradixWarning",
    "PlanError",
    "RegistryError",
    "StructureError",
    "ValidityError",
]


class GradixError(Exception):
    """Base class of gradix errors.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """

    def __init__(self, message: str, *, fix: str | None = None) -> None:
        self.message = message
        self.fix = fix
        super().__init__(message if fix is None else f"{message}\n  fix: {fix}")


class StructureError(GradixError, ValueError):
    """Inputs have the wrong structure: shapes, ranks, dtypes, field kinds or missing parts.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class BindingError(StructureError):
    """A value bound to a Pipeline by field path does not fit what the Pipeline planned.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class CapacityError(StructureError):
    """A batch or slot count exceeds a Pipeline's capacity.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class PlanError(GradixError):
    """No element can render a part of the inputs, or a planning rule failed.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class ValidityError(GradixError):
    """An element's validity predicate failed with severity ``error``.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class GradientPathError(GradixError, RuntimeError):
    """A field requires gradients, but every path from it has zero gradient.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class EnvelopeError(GradixError, ValueError):
    """Values leave the Pipeline's envelope, or an envelope entry is malformed.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """


class RegistryError(GradixError, KeyError):
    """A registry lookup or registration failed.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str or None, optional
        How to fix it; appended to the message.
    """

    def __str__(self) -> str:
        # KeyError quotes its argument; keep the message readable instead.
        return self.args[0] if self.args else ""


class GradixWarning(UserWarning):
    """Warnings from validity checks (severity ``warn``) and other non-fatal findings."""
