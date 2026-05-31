"""Deprecation helpers for legacy console script entry points."""

from __future__ import annotations

import warnings


def warn_deprecated(*, legacy_script: str, replacement: str) -> None:
    """Emit a deprecation warning for a legacy ``drevalpy-*`` console script."""
    warnings.warn(
        f"{legacy_script} is deprecated; use `{replacement}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
