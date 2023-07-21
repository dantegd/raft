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

#include <raft/neighbors/cagra.cuh>
#include <raft_runtime/neighbors/cagra.hpp>

namespace raft::runtime::neighbors::experimental::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                                         \
  void build_knn_graph(raft::resources const& handle,                                          \
                     mdspan<const DataT, matrix_extent<IdxT>, row_major, accessor> dataset,    \
                     raft::host_matrix_view<IdxT, IdxT, row_major> knn_graph,                  \
                     std::optional<float> refine_rate,                                         \
                     std::optional<ivf_pq::index_params> build_params,                         \
                     std::optional<ivf_pq::search_params> search_params)                       \
  {                                                                                            \
    raft::neighbors::experimental::cagra::build_knn_graph<T, IdxT>(handle,                     \
                                                                   dataset,                    \
                                                                   knn_graph,                  \
                                                                   refine_rate,                \
                                                                   build_params,               \
                                                                   search_params);             \
  }                                                                                            \
                                                                                               \
  void sort_knn_graph(raft::resources const& handle,                                           \
                      mdspan<const DataT, matrix_extent<IdxT>, row_major, d_accessor> dataset, \
                      mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph)      \
  {                                                                                            \
    raft::neighbors::experimental::cagra::sort_knn_graph<T, IdxT>(handle,                      \
                                                                  dataset,                     \
                                                                  knn_graph);                  \
  }                                                                                            \
                                                                                               \
  void optimize(raft::resources const& handle,                                                 \
                mdspan<IdxT, matrix_extent<IdxT>, row_major, g_accessor> knn_graph,            \
                raft::host_matrix_view<IdxT, IdxT, row_major> new_graph)                       \
  {                                                                                            \
    raft::neighbors::experimental::cagra::optimize<T, IdxT>(handle,                            \
                                                            knn_graph,                         \
                                                            new_graph);                        \
  }                                                                                            \
                                                                                               \
  auto build(raft::resources const& handle,                                                    \
             const raft::neighbors::experimental::cagra::index_params& params,                 \
             mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset)                \
    ->raft::neighbors::experimental::cagra::index<T, IdxT>                                     \
  {                                                                                            \
    return raft::neighbors::experimental::cagra::build<T, IdxT>(handle, params, dataset);      \
  }

RAFT_INST_CAGRA_BUILD(float, int64_t);
RAFT_INST_CAGRA_BUILD(int8_t, int64_t);
RAFT_INST_CAGRA_BUILD(uint8_t, int64_t);

#undef RAFT_INST_CAGRA_BUILD

}  // namespace raft::runtime::neighbors::experimental::cagra
