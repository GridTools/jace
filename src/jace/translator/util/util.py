"""Contains all general helper functions needed inside the translator.
"""

from typing import Union, Any

def list_to_dict(
        inp: list[Union[tuple[None, Any], tuple[Any, Any]]]
) -> dict[Any, Any]:
    """This method turns a `list` of pairs into a `dict` and applies a `None` filter.

    The function will only include pairs whose key, i.e. first element is not `None`.
    """
    return {k:v  for k, v in inp if k is not None}
# end def: ListToDict

