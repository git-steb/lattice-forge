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

inline double sgn1(double x) {
    return (x > 0) - (x < 0);
}

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

py::array_t<double> closestIndexC(py::array_t<double> H, py::array_t<double> x = py::array_t<double>()) {
    py::buffer_info H_buf = H.request();
    auto H_ptr = static_cast<double*>(H_buf.ptr);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H_map(H_ptr, H_buf.shape[0], H_buf.shape[1]);

    int n = H_map.rows();
    bool allnn = true;
    double epsilon = 0;

    double bestdist = std::numeric_limits<double>::infinity();
    int k = n;
    MatrixXd dist = MatrixXd::Zero(n, 1);

    MatrixXd e = MatrixXd::Zero(H_map.rows(), H_map.cols());
    bool compCP = x.size() == 0;
    if (!compCP) {
        py::buffer_info x_buf = x.request();
        auto x_ptr = static_cast<double*>(x_buf.ptr);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> x_map(x_ptr, 1, H_buf.shape[1]);
        e.row(k - 1) = x_map * H_map;
    } else {
        compCP = true;
    }

    MatrixXd u = MatrixXd::Zero(1, n);
    u(k - 1) = round(e(k - 1, k - 1));
    MatrixXd uhat(0, H_map.cols()), dhat(0, 1);

    double y = (e(k - 1, k - 1) - u(k - 1)) / H_map(k - 1, k - 1);
    MatrixXd step = MatrixXd::Zero(n, 1);
    step(k - 1) = sgn1(y);

    while (true) {
        double newdist = dist(k - 1) + y * y;
        if ((newdist - bestdist) < (epsilon * bestdist)) {
            if (k != 1) {
                dist(k - 1) = newdist;
                k--;
                e.row(k - 1) = e.row(k - 1) - u(k) * H_map.col(k - 1).transpose();
                u(k - 1) = round(e(k - 1, k - 1));
                y = (e(k - 1, k - 1) - u(k - 1)) / H_map(k - 1, k - 1);
                step(k - 1) = sgn1(y);
            } else {
                if (!compCP || (newdist != 0)) {
                    if (allnn) {
                        uhat.conservativeResize(uhat.rows() + 1, H_map.cols());
                        uhat.row(uhat.rows() - 1) = u;
                        dhat.conservativeResize(dhat.rows() + 1, 1);
                        dhat(dhat.rows() - 1) = newdist;
                    } else {
                        uhat = u;
                        k++;
                    }
                    bestdist = std::min(bestdist, double(newdist));
                }
                u(k - 1) = u(k - 1) + step(k - 1);
                y = (e(k - 1, k - 1) - u(k - 1)) / H_map(k - 1, k - 1);
                step(k - 1) = -step(k - 1) - sgn1(step(k - 1));
            }
        } else {
            if (k == n) {
                if (allnn) {
                    dhat /= bestdist;
                    Eigen::Array<bool, Eigen::Dynamic, 1> dsel = dhat.array() < double(1 + epsilon);
                    uhat = selectRows(uhat, dsel);
                }
                if (uhat.size() == 0) {
                    std::cerr << "Error: uhat is STILL empty." << std::endl;
                    std::cerr << "H_map:" << std::endl << H_map << std::endl;
                    std::cerr << "e:" << std::endl << e << std::endl;
                    std::cerr << "dist:" << std::endl << dist << std::endl;
                }
                return py::array_t<double>({uhat.rows(), uhat.cols()}, uhat.data());
            } else {
                k++;
                u(k - 1) = u(k - 1) + step(k - 1);
                y = (e(k - 1, k - 1) - u(k - 1)) / H_map(k - 1, k - 1);
                step(k - 1) = -step(k - 1) - sgn1(step(k - 1));
            }
        }
    }
}

PYBIND11_MODULE(closest_index_cpp, m) {
    m.def("closestIndexC", &closestIndexC, py::arg("H"), py::arg("x") = py::array_t<double>());
}
