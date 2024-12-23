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

#include <join/join_common.hpp>
#include "bucket_chain_hash_join.hpp"
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/tracking_resource_adaptor.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

using tracking_adaptor = rmm::mr::tracking_resource_adaptor<rmm::mr::device_memory_resource>;

template <typename Key, bool Nullable>
void nvbench_inner_join(nvbench::state& state,
                        nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  rmm::mr::cuda_async_memory_resource async_mr{};
  rmm::mr::set_current_device_resource(&async_mr);
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join<Key, Nullable>(state, join, false);
}

template <typename Key, bool Nullable>
void nvbench_inner_join2(nvbench::state& state,
                        nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
  rmm::mr::cuda_async_memory_resource async_mr{};
  rmm::mr::set_current_device_resource(&async_mr);
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join<Key, Nullable>(state, join, true);
}

template <typename Key, bool Nullable>
void nvbench_sort_hash_join(nvbench::state& state, nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
    // Option 2: Set as current device resource
    std::cout<<"start" << std::endl;
    rmm::mr::cuda_async_memory_resource async_mr{};
    rmm::mr::set_current_device_resource(&async_mr);
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return spark_rapids_jni::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join<Key, Nullable>(state, join, false, false);
  std::cout<<"end" << std::endl;
}

template <typename Key, bool Nullable>
void nvbench_sort_hash_join2(nvbench::state& state, nvbench::type_list<Key, nvbench::enum_type<Nullable>>)
{
    // Option 2: Set as current device resource
  rmm::mr::cuda_async_memory_resource async_mr{};
  //rmm::mr::set_current_device_resource(&async_mr);

  //auto* orig_device_resource = rmm::mr::get_current_device_resource();
  tracking_adaptor mr{async_mr, true};
  rmm::mr::set_current_device_resource(&mr);
  std::cout << "Current allocated bytes1: " << mr.get_allocated_bytes() << std::endl;
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return spark_rapids_jni::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join<Key, Nullable>(state, join, true, false);
  std::cout << "Current allocated bytes3: " << mr.get_allocated_bytes() << std::endl;
}

NVBENCH_BENCH_TYPES(nvbench_inner_join, NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("inner_join")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE2);

NVBENCH_BENCH_TYPES(nvbench_inner_join2, NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("inner_join2")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);
//
//
NVBENCH_BENCH_TYPES(nvbench_sort_hash_join, NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("inner_join_bucket")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE2);
//
//
NVBENCH_BENCH_TYPES(nvbench_sort_hash_join2, NVBENCH_TYPE_AXES(JOIN_KEY_TYPE_RANGE, JOIN_NULLABLE_RANGE))
  .set_name("inner_join_bucket2")
  .set_type_axes_names({"Key", "Nullable"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);
