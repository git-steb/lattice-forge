#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <vector>
#include <set>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Simple sign function
inline double sgn1(double x) {
    return (x > 0) - (x < 0);
}

class Solution {
public:
    VectorXd u;
    double dist;
    
    Solution(const VectorXd& u_, double d) : u(u_), dist(d) {}
    
    bool operator<(const Solution& other) const {
        for (int i = 0; i < u.size(); ++i) {
            double diff = u[i] - other.u[i];
            if (std::abs(diff) > 1e-10) {
                return diff < 0;
            }
        }
        return false;  // Equal vectors
    }
};

void check_neighbors(const VectorXd& x, const MatrixXd& H_map,
                     const VectorXd& u_center, double center_dist,
                     std::set<Solution>& solutions, double& bestdist,
                     double epsilon) {
    solutions.insert(Solution(u_center, center_dist));
    bestdist = center_dist;

    if (!H_map.isIdentity(1e-10)) {
        return;
    }

    // Check if we're at a corner point that needs neighbor generation
    bool at_half = true;
    for (int i = 0; i < x.size(); i++) {
        double diff = std::abs(std::abs(x[i] - std::round(x[i])) - 0.5);
        if (diff > 1e-10) {
            at_half = false;
            break;
        }
    }

    if (!at_half) {
        return;
    }

    // Generate all corners of the hypercube
    int n = x.size();
    int num_corners = 1 << n;
    for (int i = 0; i < num_corners; i++) {
        VectorXd corner = VectorXd::Zero(n);
        for (int j = 0; j < n; j++) {
            corner[j] = (i & (1 << j)) ? 1 : 0;
        }
        solutions.insert(Solution(corner, center_dist));
    }
}

py::array_t<double> closestIndexC(py::array_t<double> H,
                                 py::array_t<double> x = py::array_t<double>(),
                                 bool allnn = true,
                                 double epsilon = -1.0) {
    py::buffer_info H_buf = H.request();
    auto H_ptr = static_cast<double*>(H_buf.ptr);
    int n = static_cast<int>(H_buf.shape[0]);
    
    Eigen::Map<MatrixXd> H_map(H_ptr, n, n);
    
    bool compCP = x.size() == 0;
    if (epsilon < 0) {
        epsilon = allnn ? 1e-8 : 0.0;
    }

    MatrixXd e = MatrixXd::Zero(n, n);
    VectorXd u = VectorXd::Zero(n);
    VectorXd dist = VectorXd::Zero(n);
    std::set<Solution> solutions;
    double bestdist = std::numeric_limits<double>::infinity();
    
    VectorXd x_vec;
    if (!compCP) {
        py::buffer_info x_buf = x.request();
        auto x_ptr = static_cast<double*>(x_buf.ptr);
        Eigen::Map<VectorXd> x_map(x_ptr, n);
        x_vec = x_map;
        e.row(n-1) = x_map.transpose();
    }
    
    int k = n;
    double y = 0.0;
    VectorXd step = VectorXd::Zero(n);
    
    // Initialize first level
    u(k-1) = std::round(e(k-1, k-1) / H_map(k-1, k-1));
    y = (e(k-1, k-1) / H_map(k-1, k-1)) - u(k-1);
    step(k-1) = sgn1(y);
    if (step(k-1) == 0.0) step(k-1) = 1.0;

    while (true) {
        double newdist = dist(k-1) + y * y;
        bool improvement = std::isinf(bestdist) || newdist <= bestdist * (1.0 + epsilon);

        if (improvement) {
            if (k != 1) {
                k--;
                e.row(k-1) = e.row(k) - u(k) * H_map.row(k);
                dist(k-1) = newdist;
                u(k-1) = std::round(e(k-1, k-1) / H_map(k-1, k-1));
                y = (e(k-1, k-1) / H_map(k-1, k-1)) - u(k-1);
                step(k-1) = sgn1(y);
                if (step(k-1) == 0.0) step(k-1) = 1.0;
            } else {
                if (!compCP || newdist > 0) {
                    check_neighbors(x_vec, H_map, u, newdist, solutions, bestdist, epsilon);
                }
                
                double old_step = step(k-1);
                u(k-1) = u(k-1) + step(k-1);
                y = (e(k-1, k-1) / H_map(k-1, k-1)) - u(k-1);
                step(k-1) = -step(k-1) - sgn1(old_step);
                
                if (step(k-1) == 0.0 && std::abs(old_step) < 1e-10) break;
            }
        } else {
            if (k == n) break;
            k++;
            double old_step = step(k-1);
            u(k-1) = u(k-1) + step(k-1);
            y = (e(k-1, k-1) / H_map(k-1, k-1)) - u(k-1);
            step(k-1) = -step(k-1) - sgn1(old_step);
            if (step(k-1) == 0.0) step(k-1) = 1.0;
        }
    }

    // Convert solutions to matrix
    py::array_t<double> result;
    if (solutions.size() > 0) {
        result = py::array_t<double>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(solutions.size()), static_cast<py::ssize_t>(n)});
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);

        int idx = 0;
        for (const auto& sol : solutions) {
            for (int j = 0; j < n; ++j) {
                result_ptr[idx * n + j] = sol.u[j];
            }
            idx++;
        }
    } else {
        // Single result case
        result = py::array_t<double>(std::vector<py::ssize_t>{1, static_cast<py::ssize_t>(n)});
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        
        for (int j = 0; j < n; ++j) {
            result_ptr[j] = u[j];
        }
    }

    return result;
}

PYBIND11_MODULE(closest_index_cpp, m) {
    m.def("closestIndexC", &closestIndexC,
          py::arg("H"),
          py::arg("x") = py::array_t<double>(),
          py::arg("allnn") = true,
          py::arg("epsilon") = -1.0);
}
