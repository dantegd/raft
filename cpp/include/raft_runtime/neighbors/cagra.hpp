/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <string>

namespace raft::neighbors::experimental::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT, Accessor)                                                         \
  void build_knn_graph(raft::resources const& handle,                                          \
                       mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,                \
                       raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,                \
                       std::optional<float> refine_rate,                                       \
                       std::optional<ivf_pq::index_params> build_params,                       \
                       std::optional<ivf_pq::search_params> search_params);                    \
                                                                                               \
  void sort_knn_graph(raft::resources const& handle,                                           \
                      mdspan<const T, matrix_extent<IdxT>, row_major> dataset,                 \
                      mdspan<IdxT, matrix_extent<IdxT>, row_major> knn_graph);                 \

RAFT_INST_CAGRA_BUILD(float, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>);
RAFT_INST_CAGRA_BUILD(int8_t, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>);
RAFT_INST_CAGRA_BUILD(uint8_t, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<DataT>, memory_type::device>);

RAFT_INST_CAGRA_BUILD(float, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>);
RAFT_INST_CAGRA_BUILD(int8_t, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>);
RAFT_INST_CAGRA_BUILD(uint8_t, uint32_t,
                      host_device_accessor<std::experimental::default_accessor<IdxT>, memory_type::host>>);

#undef RAFT_INST_CAGRA_BUILD

#define RAFT_INST_CAGRA_SORT_OPT(T, IdxT)
  void optimize(raft::resources const& handle,                                                 \
                mdspan<IdxT, matrix_extent<IdxT>, row_major> knn_graph,            \
                raft::host_matrix_view<IdxT, IdxT, row_major> new_graph);                      \
                                                                                               \
  auto build(raft::resources const& handle,                                                    \
             const raft::neighbors::experimental::cagra::index_params& params,                 \
             mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset)                \
    ->raft::neighbors::experimental::cagra::index<T, IdxT>;                                    \

RAFT_INST_CAGRA_SORT_OPT(float, int64_t);
RAFT_INST_CAGRA_SORT_OPT(int8_t, int64_t);
RAFT_INST_CAGRA_SORT_OPT(uint8_t, int64_t);

#undef RAFT_INST_CAGRA_SORT_OPT

#define RAFT_INST_CAGRA_SEARCH(T, IdxT)                                                        \
  void search(raft::resources const& handle,                                                   \
              raft::neighbors::experimental::cagra::search_params const& params,            \
              const raft::neighbors::experimental::cagra::index<T, IdxT>& index,            \
              raft::device_matrix_view<const T, IdxT, row_major> queries,                      \
              raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,                       \
              raft::device_matrix_view<float, IdxT, row_major> distances);

RAFT_INST_CAGRA_SEARCH(float, int64_t);
RAFT_INST_CAGRA_SEARCH(int8_t, int64_t);
RAFT_INST_CAGRA_SEARCH(uint8_t, int64_t);

#undef RAFT_INST_CAGRA_SEARCH

#define RAFT_INST_SERIALIZE(DTYPE)                                                               \
  void serialize_file(raft::resources const& handle,                                             \
                      const std::string& filename,                                               \
                      const raft::neighbors::experimental::cagra::index<DTYPE, uint32_t>& index); \
                                                                                                 \
  void deserialize_file(raft::resources const& handle,                                           \
                        const std::string& filename,                                             \
                        raft::neighbors::experimental::cagra::index<DTYPE, uint32_t>* index);     \
  void serialize(raft::resources const& handle,                                                  \
                 std::string& str,                                                               \
                 const raft::neighbors::experimental::cagra::index<DTYPE, uint32_t>& index);      \
                                                                                                 \
  void deserialize(raft::resources const& handle,                                                \
                   const std::string& str,                                                       \
                   raft::neighbors::experimental::cagra::index<DTYPE, uint32_t>* index);          \

RAFT_INST_SERIALIZE(float);
RAFT_INST_SERIALIZE(int8_t);
RAFT_INST_SERIALIZE(uint8_t);

}  // namespace raft::runtime::neighbors::experimental::cagra
