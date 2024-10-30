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

namespace spark_rapids_jni {
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
}
