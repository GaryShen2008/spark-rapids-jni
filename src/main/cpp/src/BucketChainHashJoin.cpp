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
}  // extern "C"
