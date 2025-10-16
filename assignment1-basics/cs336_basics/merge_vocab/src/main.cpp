#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

void add_value(py::dict &d, std::string k, const int value) {
    auto k_bytes = py::bytes{k};
    int current = d.contains(k_bytes) ? d[k_bytes].cast<int>() : 0;
    d[k_bytes] = current + value;
}

void merge_(py::dict freq_table,
            py::dict new_freq_table,
            py::dict byte_pairs,
            py::dict pair_relations,
            py::bytes merge_key_bytes,
            int freq_table_size) {
    auto merge_key = merge_key_bytes.cast<std::string>();
    int count = 1;
    int c2 = 0;
    for (auto item : freq_table) {
        auto key = item.first.cast<std::vector<std::string>>();
        auto value = item.second.cast<int>();
        bool skip_next = false;
        bool merge_last = false;
        std::vector<py::bytes> new_key;
        auto n = key.size();
        for (int i = 0; i < n; i++) {
            if (skip_next) {
                skip_next = false;
                continue;
            }
            if (i == n - 1) {
                new_key.push_back(py::bytes{key[i]});
                break;
            }
            std::string pair = key[i] + key[i + 1];
            std::string left;
            if (!pair.compare(merge_key)) {
                new_key.push_back(py::bytes{pair});
                if (i > 0) {
                    if (!merge_last) {
                        auto k = (key[i - 1] + key[i]);

                        add_value(byte_pairs, k, -value);
                        left = key[i - 1];
                    } else {
                        left = key[i - 2] + key[i - 1];
                    }
                    auto new_pair = left + pair;
                    pair_relations[py::bytes{new_pair}]
                        = py::make_tuple(py::bytes{left}, py::bytes{pair});
                    add_value(byte_pairs, new_pair, value);
                }
                if (i < n - 2) {
                    bool merge_next;
                    std::string right;
                    if ((i == n - 3)
                        || (merge_key.compare(key[i + 2] + key[i + 3]) != 0)) {
                        merge_next = false;
                        right = key[i + 2];
                    } else {
                        merge_next = true;
                        right = key[i + 2] + key[i + 3];
                    }
                    auto new_pair = pair + right;

                    auto k = key[i + 1] + key[i + 2];
                    add_value(byte_pairs, k, -value);

                    pair_relations[py::bytes{new_pair.c_str()}]
                        = py::make_tuple(py::bytes{pair}, py::bytes{right});
                    if (!merge_next) {
                        add_value(byte_pairs, new_pair, value);
                    }
                }
                skip_next = true;
                merge_last = true;
            } else {
                new_key.push_back(py::bytes{key[i]});
                skip_next = false;
                merge_last = false;
            }
        }

        new_freq_table[py::tuple(py::cast(new_key))] = value;
        if (count++ == freq_table_size) {
            break;
        }
    }
}

PYBIND11_MODULE(merge_vocab, m) {
    m.def("merge", &merge_);
}