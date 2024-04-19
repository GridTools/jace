# JaCe - JAX Just-In-Time compilation using DaCe (Data Centric Parallel Programming)
#
# Copyright (c) 2024, ETH Zurich
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final


class RevisionCounterManager:
    """This class acts as a manager for revision counters.

    It is intended as a shared object and each new object that needs a revision,
    simply calls 'assign_revision()' to get the new one.
    """

    __slots__ = ("_next_revision",)

    """The revision value of the very first call to 'assign_revision()'.
    This revision is only assigned once."""
    ROOT_REVISION: Final[int] = 0

    def __init__(self) -> None:
        """Creates a revision counter manager."""
        self._next_revision = self.ROOT_REVISION

    def assign_revision(self) -> int:
        """Returns a revision number and advance self."""
        ret = self._next_revision
        self._next_revision += 1
        return ret

    def _reset_state(self) -> RevisionCounterManager:
        """This function sets the revision counter back.

        Notes:
            Calling this function is almost always an error.
            This function does not restore the state right after initialization, but one call after 'assign_revision()'.
                This is done to ensure that there is one single initial revision.
        """
        self._next_revision = self.ROOT_REVISION
        _ = self.assign_revision()  # Ensure that we throw away the root
        return self

    def is_root_revision(
        self,
        rev: int,
    ) -> bool:
        """This function checks if 'rev' revers to the (absolute) unique revision of the root."""
        return rev == self.ROOT_REVISION
