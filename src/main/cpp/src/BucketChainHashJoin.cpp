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

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BucketChainHashJoin_innerJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  try {
    cudf::jni::auto_set_device(env);
    auto const left_table = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_table = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::join_gather_maps(left_table, right_table, compare_nulls_equal));
  }
  CATCH_STD(env, 0);
}
}  // extern "C"
