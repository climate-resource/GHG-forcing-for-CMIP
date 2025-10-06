"""
Module including helper functions
"""


def ensure_trailing_slash(path: str) -> str:
    """
    Ensure trailing slash at the end of a path

    Parameters
    ----------
    path
        the path / directory

    Returns
    -------
    :
        path with trailing slash
    """
    return path if path.endswith("/") else path + "/"
