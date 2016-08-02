"""Util methods."""


class PrioItem:
    """Item storable in PriorityQueue."""

    def __init__(self, prio, data):
        """Init."""
        self.prio = prio
        self.data = data

    def __lt__(self, b):
        """lt comparison."""
        return self.prio < b.prio
