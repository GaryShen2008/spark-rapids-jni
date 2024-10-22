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

#include "cudf_jni_apis.hpp"
#include "bucket_chain_hash_join.hpp"

#include <rmm/device_buffer.hpp>
#include <iostream>
#include <cudf/copying.hpp>

using cudf::jni::ptr_as_jlong;
using cudf::jni::release_as_jlong;

namespace rapids {
namespace jni {

// Convert a cudf gather map pair into the form that Java expects
// The resulting Java long array contains the following at each index:
//   0: Size of each gather map in bytes
//   1: Device address of the gather map for the left table
//   2: Host address of the rmm::device_buffer instance that owns the left gather map data
//   3: Device address of the gather map for the right table
//   4: Host address of the rmm::device_buffer instance that owns the right gather map data
jlongArray gather_maps_to_java(
    JNIEnv* env,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> maps)
{
  // release the underlying device buffer to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(maps.first->release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(maps.second->release());

  cudf::jni::native_jlongArray result(env, 5);
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = ptr_as_jlong(left_map_buffer->data());
  result[2] = release_as_jlong(left_map_buffer);
  result[3] = ptr_as_jlong(right_map_buffer->data());
  result[4] = release_as_jlong(right_map_buffer);

  return result.get_jArray();
}

// Generate gather maps needed to manifest the result of an equi-join between two tables.
template <typename T>
jlongArray join_gather_maps(
  JNIEnv* env, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal, T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_keys, "right_table is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto nulleq = compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_maps_to_java(env, join_func(*left_keys, *right_keys, nulleq));
  }
  CATCH_STD(env, NULL);
}


jlongArray convert_table_for_return(JNIEnv* env,
                                    std::unique_ptr<cudf::table>&& table_result,
                                    std::vector<std::unique_ptr<cudf::column>>&& extra_columns)
{
  std::vector<std::unique_ptr<cudf::column>> ret = table_result->release();
  int table_cols                                 = ret.size();
  int num_columns                                = table_cols + extra_columns.size();
  cudf::jni::native_jlongArray outcol_handles(env, num_columns);
  std::transform(ret.begin(), ret.end(), outcol_handles.begin(), [](auto& col) {
    return release_as_jlong(col);
  });
  std::transform(
    extra_columns.begin(), extra_columns.end(), outcol_handles.begin() + table_cols, [](auto& col) {
      return release_as_jlong(col);
    });
  return outcol_handles.get_jArray();
}


} // jni
} // rapids
extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_BucketChainHashJoin_innerJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
    try{
        return rapids::jni::join_gather_maps(
            env,
            j_left_keys,
            j_right_keys,
            compare_nulls_equal,
            [](cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
                return spark_rapids_jni::inner_join(left, right, nulleq);
        });
    }
    CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_BucketChainHashJoin_gather(
  JNIEnv* env, jclass, jlong j_input1, jlong j_input2, jlong j_map1, jlong j_map2, jboolean check_bounds)
{
  JNI_NULL_CHECK(env, j_input1, "input table is null", 0);
  JNI_NULL_CHECK(env, j_map1, "map column is null", 0);
  JNI_NULL_CHECK(env, j_input2, "input table is null", 0);
  JNI_NULL_CHECK(env, j_map2, "map column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input1 = reinterpret_cast<cudf::table_view const*>(j_input1);
    auto const map1   = reinterpret_cast<cudf::column_view const*>(j_map1);
    auto const input2 = reinterpret_cast<cudf::table_view const*>(j_input2);
    auto const map2   = reinterpret_cast<cudf::column_view const*>(j_map2);
    auto bounds_policy =
      check_bounds ? cudf::out_of_bounds_policy::NULLIFY : cudf::out_of_bounds_policy::DONT_CHECK;
    return rapids::jni::convert_table_for_return(env, spark_rapids_jni::gather(*input1, *input2, *map1, *map2, bounds_policy), {});
  }
  CATCH_STD(env, 0);
}


}  // extern "C"
