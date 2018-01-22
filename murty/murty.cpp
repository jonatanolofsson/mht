#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include "murty.hpp"

namespace py = pybind11;

PYBIND11_MODULE(murty, m) {
    py::class_<lap::Murty>(m, "Murty")
        .def(py::init<lap::CostMatrix>())
        .def("draw", &lap::Murty::draw_tuple);
}
