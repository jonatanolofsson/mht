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
from lapjv import lap

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
    try:
        Q = queue.PriorityQueue()
        M = C.shape[0]
        N = C.shape[1]
        cost, assign = lap(C)[0:2]
        Q.put((cost, list(assign),
               (), (),
               (), ()))
        k = 0
        while not Q.empty():
            S = Q.get_nowait()
            yield (S[0], S[1][:M])
            k += 1
            ni = len(S[2])

            rmap = tuple(x for x in range(M) if x not in S[2])
            cmap = tuple(x for x in S[1] if x not in S[3])
            cmap += tuple(x for x in range(N)
                          if x not in S[3] and x not in S[1])

            removed_values = C[S[4], S[5]]
            C[S[4], S[5]] = LARGE

            C_ = C[rmap, :][:, cmap]
            for t in range(M - ni):
                removed_value = C_[t, t]
                C_[t, t] = LARGE

                cost, lassign = lap(C_[t:, t:])[0:2]
                if LARGE not in C_[range(t, t + len(lassign)), lassign + t]:
                    cost += C[S[2], S[3]].sum()
                    cost += C_[range(t), range(t)].sum()
                    assign = [None] * M
                    for r in range(ni):
                        assign[S[2][r]] = S[3][r]
                    for r in range(t):
                        assign[rmap[r]] = cmap[r]
                    for r in range(len(lassign)):
                        assign[rmap[r + t]] = cmap[lassign[r] + t]

                    nxt = (cost, assign,
                           S[2] + tuple(rmap[x] for x in range(t)),
                           S[3] + tuple(cmap[:t]),
                           S[4] + (rmap[t],),
                           S[5] + (cmap[t],))
                    Q.put(nxt)
                C_[t, t] = removed_value
            C[S[4], S[5]] = removed_values
    except GeneratorExit:
        pass
