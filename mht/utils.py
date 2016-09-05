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


def anyitem(iterable):
    """Retrieve 'first' item from set."""
    try:
        return next(iter(iterable))
    except StopIteration:
        return None


def connected_components(connections):
    """Get all connected components."""
    seen = set()

    def component(node):
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= connections[node] - seen
            yield node
    for node in list(connections.keys()):
        if node not in seen:
            yield set(component(node))
