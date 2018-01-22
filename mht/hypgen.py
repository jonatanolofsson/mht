"""Hypothesis generation."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import queue
from copy import copy
import murty as murty_

from .utils import LARGE


def permgen(lists, presorted=False):
    """Generate ordered permutations of lists of (cost, data) tuples."""
    if not presorted:
        lists = [sorted(l) for l in lists]
    bounds = [len(l) - 1 for l in lists]
    N = len(lists)
    Q = queue.PriorityQueue()
    Q.put((0, [0] * N))
    prev_cost = 1
    prev_states = []
    while not Q.empty():
        cost, state = Q.get_nowait()
        if cost == prev_cost:
            if state in prev_states:
                continue
        else:
            prev_states = []
        prev_states.append(state)
        prev_cost = cost
        for n in range(N):
            if state[n] < bounds[n]:
                nstate = copy(state)
                nstate[n] += 1
                ncost = sum(l[nstate[i]][0] for i, l in enumerate(lists))
                Q.put((ncost, nstate))
        yield ([l[state[n]][1] for n, l in enumerate(lists)],
               None if Q.empty() else Q.queue[0][0])


def murty(C):
    """Algorithm due to Murty."""
    mgen = murty_.Murty(C)
    while True:
        ok, cost, sol = mgen.draw()
        if not ok:
            return None
        yield cost, sol
