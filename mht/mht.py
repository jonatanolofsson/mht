"""Library implementing Multiple Hypothesis assigning."""

import queue
from lapjv import lap


def murty(C):
    """Algorithm due to Murty."""
    try:
        Q = queue.PriorityQueue()
        N = C.shape[1]
        M = C.shape[0]
        if N != M:
            C = C.copy()
            C.resize(N, N)
        large_value = 100000
        cost, assign = lap(C)[0:2]
        Q.put((cost, assign,
               (), (),
               tuple(range(len(assign))), tuple(assign),
               (), ()))
        k = 0
        while not Q.empty():
            S = Q.get_nowait()
            yield (S[0], S[1][:M])
            k += 1

            rmap = S[4] + S[6]
            cmap = S[5] + S[7]
            C_ = C[rmap, :][:, cmap]

            rdiag = tuple(range(len(cmap) - len(S[6]), len(cmap)))
            old_diag = C_[rdiag, rdiag]
            C_[rdiag, rdiag] = large_value

            ni = len(S[2])
            nd = len(S[6])

            for i in range(N - nd - ni - 1):
                old_value = C_[i, i]
                C_[i, i] = large_value

                cost, lassign = lap(C_[i:, i:])[0:2]
                cost += C[S[2], S[3]].sum()
                cost += C_[range(i), range(i)].sum()
                assign = [None] * N
                for r in range(ni):
                    assign[S[2][r]] = S[3][r]
                for r in range(i):
                    assign[rmap[r]] = cmap[r]
                for r in range(len(lassign)):
                    assign[rmap[r + i]] = cmap[lassign[r] + i]

                nxt = (cost, assign,
                       S[2] + (rmap[:i]),
                       S[3] + (cmap[:i]),
                       S[4][i + 1:],
                       S[5][i + 1:],
                       S[6] + (rmap[i],),
                       S[7] + (cmap[i],))
                Q.put(nxt)

                C_[i, i] = old_value
            C_[rdiag, rdiag] = old_diag
    except GeneratorExit:
        pass


class Measurement:
    """Class for containing measurement."""

    def __init__(self, z, R, mfn):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn

    def __str__(self):
        """Return string representation of measurement."""
        return "M({}, {})".format(self.z, self.R)
