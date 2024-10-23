"""
Module ``util`` has been deprecated.
"""
import warnings

warnings.warn(
    "Module 'drizzle.util' has been deprecated since version 2.0.0 "
    "and it will be removed in a future release. "
    "Please replace calls to 'util.is_blank()' with alternative "
    "implementation.",
    DeprecationWarning
)


def is_blank(value):
    """
    Determines whether or not a value is considered 'blank'.

    Parameters
    ----------
    value : str
        The value to check

    Returns
    -------
    True or False
    """
    warnings.warn(
        "'is_blank()' has been deprecated since version 2.0.0 "
        "and it will be removed in a future release. "
        "Please replace calls to 'is_blank()' with alternative implementation.",
        DeprecationWarning
    )
    return value.strip() == ""
