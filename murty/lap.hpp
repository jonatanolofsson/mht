// Copyright 2018 Jonatan Olofsson
#pragma once

#include <Eigen/Core>

namespace lap {

using Assignment = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Slack = Eigen::Matrix<double, Eigen::Dynamic, 1>;
static const int inf = 300000;

template<typename CostMatrix, typename Assignment, typename RowSlack, typename ColSlack>  // NOLINT
void lap(const CostMatrix& C, Assignment& x, RowSlack& u, ColSlack& v) {
    int N = C.rows();
    int M = C.cols();
    int cnt, H, f, f0, i, i1, j, j1, j2 = 0, k, last = 0, low, min = 0, up;
    double h, u1, u2;
    Eigen::RowVectorXi free(N), col(M), y(M), pred(M);
    Eigen::RowVectorXd d(M);

    x.setConstant(-1);
    y.setConstant(-1);
    v.setZero();
    u.setZero();

    for (i = 0; i < N; ++i) { free[i] = i; }
    for (j = 0; j < M; ++j) { col[j] = j; }

    f = N;  // Each row is still unassigned

    for (cnt = 0; cnt < 2; ++cnt) {
        k = 0;
        f0 = f;
        f = 0;
        while (k < f0) {
            i = free[k++];
            u1 = C(i, 0) - v[0];
            j1 = 0;
            u2 = inf;
            for (j = 1; j < M; ++j) {
                h = C(i, j) - v[j];
                if (h < u2) {
                    if (h >= u1) {
                        u2 = h;
                        j2 = j;
                    } else {
                        u2 = u1;
                        u1 = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            i1 = y[j1];
            if (u1 < u2) {
                v[j1] = v[j1] - u2 + u1;
            } else if (i1 >= 0) {
                j1 = j2;
                i1 = y[j1];
            }
            if (i1 >= 0) {
                if (u1 < u2) {
                    free[--k] = i1;
                    x[i1] = -1;
                } else {
                    free[f++] = i1;
                    x[i1] = -1;
                }
            }
            x[i] = j1;
            y[j1] = i;
        }
    }

    // --------------------------------------------------------
    // Augmentation:
    //

    f0 = f;
    for (f = 0; f < f0; ++f) {  // Find augmenting path for each unassigned row
        i1 = free[f];
        low = 0;
        up = 0;
        for (j = 0; j < M; ++j) {
            d[j] = C(i1, j) - v[j];
            pred[j] = i1;
        }
        while (true) {
            if (up == low) {  // Find columns with new value for minimum d
                last = low - 1;
                min = d[col[up]];
                up = up + 1;
                for (k = up; k < M; ++k) {
                    j = col[k];
                    h = d[j];
                    if (h <= min) {
                        if (h < min) {
                            up = low;
                            min = h;
                        }
                        col[k] = col[up];
                        col[up] = j;
                        up = up + 1;
                    }
                }
                for (H = low; H < up; ++H) {
                    j = col[H];
                    if (y[j] == -1) {
                        goto augment;
                    }
                }
            }  //{ up=low }
            j1 = col[low];
            low = low + 1;
            i = y[j1];
            u1 = C(i, j1) - v[j1] - min;
            for (k = up; k < M; ++k) {
                j = col[k];
                h = C(i, j) - v[j] - u1;
                if (h < d[j]) {
                    d[j] = h;
                    pred[j] = i;
                    if (h == min) {
                        if (y[j] == -1) {
                            goto augment;
                        } else {
                            col[k] = col[up];
                            col[up] = j;
                            up = up + 1;
                        }
                    }
                }
            }  // for k
        }

        augment:
        for (k = 0; k < last; ++k) {  // Updating of column prices
            j1 = col[k];
            v[j1] = v[j1] + d[j1] - min;
        }
        do {  // Augmentation
            i = pred[j];
            y[j] = i;
            k = j;
            j = x[i];
            x[i] = k;
        } while (i != i1);
    }  // { for f }

    for (i = 0; i < N; ++i) {
        j = x[i];
        u[i] = C(i, j) - v[j];
    }
}


template<typename CostMatrix>
Assignment lap(const CostMatrix& C) {
    Slack u(C.rows());
    Slack v(C.cols());
    Assignment x(C.rows());
    lap(C, x, u, v);
    return x;
}

template<typename CostMatrix>
inline double cost(CostMatrix& C, Assignment res) {
    double c = 0;
    for (unsigned i = 0; i < res.rows(); ++i) {
        c += C(i, res[i]);
    }
    return c;
}

}  // namespace lap
