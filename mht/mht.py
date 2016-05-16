"""Library implementing Multiple Hypothesis assigning."""

import queue
import numpy as np
from copy import copy
from lapjv import lap

from .target import Target


class MHT:
    """MHT class."""

    def __init__(self):
        """Init."""
        self.targets = []

    def predict(self, dT):
        """Move to next timestep."""
        for t in self.targets:
            t.predict(dT)

    def _relevant_targets(self, scan):
        """Filter out relevant targets."""
        return self.targets

    def register_scan(self, scan):
        """Register scan."""
        targets = self._relevant_targets(scan)
        for hyp in self._hypothesis_factory(targets, scan):
            for a, m in hyp:
                a.assign(m)

    def _hypothesis_factory(self, targets, scan):
        """Generate global hypotheses."""
        new_targets = {}

        def new_target(m):
            if m not in new_targets:
                obj = new_targets[m] = Target()
                self.targets.append(obj)

        def get_pgen(scan, tH):
            target_assignment = [
                (scan.reports[r],
                 targets[a] if a < N else new_target(a))
                for r, a in enumerate(tH[1]) if a < 2*N]

            return permgen([target.score(report) for
                            report, target in target_assignment])

        M = len(scan.reports)
        N = len(targets)
        C = np.zeros((M, M + 2*N))
        C[:, N:2*N] = -scan.sensor.score_new
        C[:, 2*N:] = -scan.sensor.score_false
        for t, target in enumerate(targets):
            C[:, t] = [-max(x[0] for x in target.score(r))
                       for r in scan.reports]
        target_hypgen = murty(C)

        Q = queue.PriorityQueue()
        tH = next(target_hypgen)
        nxt_tH = next(target_hypgen)
        Q.put((tH[0], get_pgen(scan, tH)))
        while not Q.empty():
            if nxt_tH and Q.queue[0][0] > nxt_tH[0]:
                tH = nxt_tH
                nxt_tH = next(target_hypgen)
                pgen = get_pgen(scan, tH)
            else:
                pgen = Q.get_nowait()[1]

            for track_assignment, next_trackcost in pgen:
                # yield track_assignment
                print(track_assignment)
                if next_trackcost > Q.queue[0][0]:
                    Q.put((next_trackcost, pgen))
                    break


def permgen(lists):
    """Generate ordered permutations of lists."""
    lists = [sorted(ol) for ol in lists]
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
        yield ([l[state[n]][1] for n, l in enumerate(lists)], Q.queue[0][0])


def murty(C):
    """Algorithm due to Murty."""
    try:
        LARGE = 10000
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


class Report:
    """Class for containing measurement."""

    def __init__(self, z, R, mfn):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn

    def __str__(self):
        """Return string representation of measurement."""
        return "R({}, {})".format(self.z, self.R)


class Scan:
    """Report container class."""

    def __init__(self, sensor, reports):
        """Init."""
        self.sensor = sensor
        self.reports = reports
