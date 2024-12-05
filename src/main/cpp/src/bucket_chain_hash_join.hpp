/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <iostream>
namespace spark_rapids_jni {

//namespace detail{

/**
 * @brief Forward declaration for our hash join
 */
//template <typename T>
//class partition_hash_join;
//
//}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
inner_join(cudf::table_view const& left_input,
           cudf::table_view const& right_input,
           cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
           rmm::cuda_stream_view stream      = cudf::get_default_stream(),
           rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());


std::unique_ptr<cudf::table> gather(cudf::table_view const& source_table,
            cudf::column_view const& gather_map,
            cudf::out_of_bounds_policy bounds_policy = cudf::out_of_bounds_policy::DONT_CHECK,
            rmm::cuda_stream_view stream = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());


//class partition_hash_join{
//
//  public:
//    partition_hash_join() = delete;
//    ~partition_hash_join();
//    partition_hash_join(partition_hash_join const&) = delete;
//    partition_hash_join(partition_hash_join&&) = delete;
//    partition_hash_join& operator=(partition_hash_join const&) = delete;
//    partition_hash_join& operator=(partition_hash_join&&) = delete;
//
//  private:
//
//    const cudf::table_view& r;
//    const cudf::table_view& s;
//    // following code was copied from cudf
//    bool const _is_empty;   ///< true if `_hash_table` is empty
//    bool const _has_nulls;  ///< true if nulls are present in either build table or any probe table
//    cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
//    cudf::table_view _build;                 ///< input table to build the hash map
////
////    int r_cols = r.num_columns();
////    int s_cols = s.num_columns();
//
//    static constexpr uint32_t log2_bucket_size = 12;
//    static constexpr uint32_t bucket_size = (1 << log2_bucket_size);
//    static constexpr int LOCAL_BUCKETS_BITS = 11;
//    static constexpr int SHUFFLE_SIZE = sizeof(key_t) == 4 ? 32 : 16;
//    int threshold = 2 * bucket_size;
//
//    int nr;
//    int ns;
//    unsigned long long int n_matches = 2;
//    int circular_buffer_size;
//    int first_bit;
//    int n_partitions;
//    int radix_bits;
//
//    // get both data type of keys to join
//    // data_type of keyR
//    // data_type of keyS
//
//    unsigned long long int*   d_n_matches     {nullptr};
//    int*   r_offsets       {nullptr};
//    int*   s_offsets       {nullptr};
//    uint64_t* r_work       {nullptr};
//    uint64_t* s_work       {nullptr};
//    int*   total_work      {nullptr};
//    key_t* rkeys_partitions {nullptr};
//    key_t* skeys_partitions {nullptr};
//    key_t* rkeys_partitions_tmp {nullptr};
//    key_t* skeys_partitions_tmp {nullptr};
//    key_t*  rvals_partitions {nullptr};
//    key_t*  svals_partitions {nullptr};
//    int*   r_match_idx     {nullptr};
//    int*   s_match_idx     {nullptr};
//
//  public:
//
////    std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
////              std::unique_ptr<rmm::device_uvector<size_type>>>
////    join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
//
//    partition_hash_join(cudf::table_view &r_in, cudf::table_view &s_in, int first_bit, int radix_bits, int circular_buffer_size, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
//    partition_hash_join(cudf::table_view &r_in, cudf::table_view &s_in, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
//};


}
