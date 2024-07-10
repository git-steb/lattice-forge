#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <limits>

namespace py = pybind11;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Utility function for sign
inline double sgn1(double x) {
    return (x > 0) - (x < 0);
}

// Function to select rows based on boolean array
inline Eigen::MatrixXd selectRows(const Eigen::MatrixXd& m, const Eigen::Array<bool, Eigen::Dynamic, 1>& v) {
    int n = v.count();
    Eigen::MatrixXd r(n, m.cols());
    int k = 0;
    for (int i = 0; i < v.size(); i++) {
        if (v(i)) {
            r.row(k++) = m.row(i);
        }
    }
    return r;
}

Eigen::MatrixXd closestIndexC(const Eigen::MatrixXd& H, py::array_t<double> x = py::array_t<double>()) {
    int n = H.rows();
    bool allnn = true; // compute all nearest neighbours?
    double epsilon = 0;

    double bestdist = std::numeric_limits<double>::infinity();
    int k = n;
    MatrixXd dist = MatrixXd::Zero(n, 1);

    MatrixXd e = MatrixXd::Zero(H.rows(), H.cols());
    bool compCP = x.size() == 0;
    if (!compCP) {
        auto buf = x.request();
        double* x_ptr = static_cast<double*>(buf.ptr);
        Eigen::Map<MatrixXd> x_map(x_ptr, 1, H.cols());
        e.row(k - 1) = x_map * H; // all indexing off by one from MATLAB
    } else {
        compCP = true; // second arg may be present, but empty
    }

    MatrixXd u = MatrixXd::Zero(1, n);
    u(k - 1) = round(e(k - 1, k - 1));
    MatrixXd uhat, dhat;

    double y = (e(k - 1, k - 1) - u(k - 1)) / H(k - 1, k - 1);
    MatrixXd step = MatrixXd::Zero(n, 1);
    step(k - 1) = sgn1(y);

    while (true) {
        double newdist = dist(k - 1) + y * y;
        if ((newdist - bestdist) < (epsilon * bestdist)) {
            if (k != 1) {
                e.block(k - 2, 0, 1, k - 1) = e.block(k - 1, 0, 1, k - 1) - y * H.block(k - 1, 0, 1, k - 1);
                k--;
                dist(k - 1) = newdist;
                double ekk = e(k - 1, k - 1);
                u(k - 1) = round(ekk); // closest layer
                y = (ekk - u(k - 1)) / H(k - 1, k - 1);
                step(k - 1) = sgn1(y);
            } else {
                if (!compCP || (newdist != 0)) {
                    if (allnn) {
                        uhat.conservativeResize(uhat.rows() + 1, H.cols());
                        uhat.row(uhat.rows() - 1) = u;
                        dhat.conservativeResize(dhat.rows() + 1, 1);
                        dhat(dhat.rows() - 1) = newdist;
                    } else { // only keep closest point
                        uhat = u;
                        k++;
                    }
                    bestdist = std::min(bestdist, double(newdist));
                }
                u(k - 1) = u(k - 1) + step(k - 1);
                y = (e(k - 1, k - 1) - u(k - 1)) / H(k - 1, k - 1);
                step(k - 1) = -step(k - 1) - sgn1(step(k - 1));
            }
        } else {
            if (k == n) {
                if (allnn) {
                    dhat /= bestdist;
                    Eigen::Array<bool, Eigen::Dynamic, 1> dsel = dhat.array() < double(1 + epsilon);
                    uhat = selectRows(uhat, dsel);
                }
                return uhat;
            } else {
                k++;
                u(k - 1) = u(k - 1) + step(k - 1);
                y = (e(k - 1, k - 1) - u(k - 1)) / H(k - 1, k - 1);
                step(k - 1) = -step(k - 1) - sgn1(step(k - 1));
            }
        }
    }
}

PYBIND11_MODULE(closest_index, m) {
    m.def("closestIndexC", &closestIndexC, py::arg("H"), py::arg("x") = py::array_t<double>());
}
