#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mol1x2.h"

namespace py = pybind11;

// 定义CombinedConformer的Python绑定
struct PyCombinedConformer {
    float total_energy;
    float energy1;
    float energy2;
    float interaction_energy;
    int rot_angle;
    int s1_index;
    int s2_index;
    int nn_atom;
    std::vector<int> zi;
    std::vector<float> zb;
    std::string energy_line;
    
    PyCombinedConformer() = default;
    
    PyCombinedConformer(const CombinedConformer& c) :
        total_energy(c.total_energy),
        energy1(c.energy1),
        energy2(c.energy2),
        interaction_energy(c.interaction_energy),
        rot_angle(c.rot_angle),
        s1_index(c.s1_index),
        s2_index(c.s2_index),
        nn_atom(c.nn_atom),
        zi(c.zi, c.zi + c.nn_atom),
        zb(c.zb, c.zb + 3 * c.nn_atom),
        energy_line(c.energy_line) {}
};

// 定义CombinedResult的Python绑定
struct PyCombinedResult {
    int n_conformers;
    std::vector<PyCombinedConformer> conformers;
    
    PyCombinedResult() : n_conformers(0) {}
    
    PyCombinedResult(const CombinedResult& r) :
        n_conformers(r.n_conformers),
        conformers(r.n_conformers) {
        for (int i = 0; i < r.n_conformers; i++) {
            conformers[i] = PyCombinedConformer(r.conformers[i]);
        }
    }
};

PYBIND11_MODULE(mol1x2, m) {
    m.doc() = "Python bindings for mol1x2 molecular structure combination tool";
    
    // 绑定XYZSet包装类
    py::class_<XYZSet>(m, "XYZSet")
        .def(py::init([]() {
            XYZSet* set = create_xyz_set();
            return set;
        }))
        .def("__del__", [](XYZSet& self) {
            free_xyz_set(&self);
        })
        .def_property_readonly("nn_atom", [](XYZSet& self) {
            return get_xyz_set_nn_atom(&self);
        })
        .def_property_readonly("stn", [](XYZSet& self) {
            return get_xyz_set_stn(&self);
        })
        .def_property_readonly("mine", [](XYZSet& self) {
            return get_xyz_set_mine(&self);
        });
    
    // 绑定CombinedConformer结构体
    py::class_<PyCombinedConformer>(m, "CombinedConformer")
        .def(py::init<>())
        .def_readonly("total_energy", &PyCombinedConformer::total_energy)
        .def_readonly("energy1", &PyCombinedConformer::energy1)
        .def_readonly("energy2", &PyCombinedConformer::energy2)
        .def_readonly("interaction_energy", &PyCombinedConformer::interaction_energy)
        .def_readonly("rot_angle", &PyCombinedConformer::rot_angle)
        .def_readonly("s1_index", &PyCombinedConformer::s1_index)
        .def_readonly("s2_index", &PyCombinedConformer::s2_index)
        .def_readonly("nn_atom", &PyCombinedConformer::nn_atom)
        .def_readonly("zi", &PyCombinedConformer::zi)
        .def_readonly("zb", &PyCombinedConformer::zb)
        .def_readonly("energy_line", &PyCombinedConformer::energy_line);
    
    // 绑定CombinedResult结构体
    py::class_<PyCombinedResult>(m, "CombinedResult")
        .def(py::init<>())
        .def_readonly("n_conformers", &PyCombinedResult::n_conformers)
        .def_readonly("conformers", &PyCombinedResult::conformers);
    
    // 绑定全局阈值函数
    m.def("set_global_thresholds", &set_global_thresholds, "Set global energy thresholds");
    m.def("get_global_threshold_e0", &get_global_threshold_e0, "Get global energy threshold E0");
    
    // 绑定主要功能函数
    m.def("init_element_table", &export_init_element_table, "Initialize element table");
    m.def("elem_symbol_to_Z", &export_elem_symbol_to_Z, "Convert element symbol to atomic number");
    m.def("read_xyz_all", [](const std::string& fname, XYZSet& out) {
        return export_read_xyz_all(fname.c_str(), &out);
    }, "Read XYZ file");
    
    m.def("read_xyz_from_string", [](const std::string& content, XYZSet& out) {
        return export_read_xyz_from_string(content.c_str(), &out);
    }, "Read XYZ from string content");
    
    m.def("read_xyz_from_string_with_validation", [](const std::string& content, XYZSet& out) {
        char error_msg[256] = {0};
        int result = export_read_xyz_from_string_with_validation(content.c_str(), &out, error_msg, sizeof(error_msg));
        if (result) {
            return py::make_tuple(result, "");
        } else {
            return py::make_tuple(result, std::string(error_msg));
        }
    }, "Read XYZ from string content with validation");
    
    m.def("rpip_1x2m", [](const XYZSet& A, const XYZSet& B,
                         int mol1_abond_1, int mol1_abond_2,
                         int mol2_abond_1, int mol2_abond_2,
                         int max_add_outn) {
        CombinedResult result = export_rpip_1x2m(&A, &B, mol1_abond_1, mol1_abond_2,
                                         mol2_abond_1, mol2_abond_2, max_add_outn);
        PyCombinedResult py_result(result);
        export_free_combined_result(&result);
        return py_result;
    }, "Combine two molecular structures");
}