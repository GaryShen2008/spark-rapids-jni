/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>
#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

extern "C" {

/*
 * Class:     com_nvidia_spark_rapids_jni_CudaIPC
 * Method:    getCudaIpcMemHandleBytes
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_CudaIPC_getCudaIpcMemHandleBytes
(JNIEnv *jenv, jclass jclz, jlong jdev_ptr)
{
  cudaIpcMemHandle_t handle;
  CUDF_CUDA_TRY(cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(jdev_ptr)));
	jbyteArray jout = jenv->NewByteArray(sizeof(cudaIpcMemHandle_t));
  jenv->SetByteArrayRegion(jout, 0, sizeof(cudaIpcMemHandle_t), reinterpret_cast<const jbyte *>(&handle));
  return jout;
}

/*
 * Class:     com_nvidia_spark_rapids_jni_CudaIPC
 * Method:    cudaDeviceSynchronize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_CudaIPC_cudaDeviceSynchronize
(JNIEnv *jenv, jclass jclz)
{
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

/*
 * Class:     com_nvidia_spark_rapids_jni_CudaIPC
 * Method:    checkGpuMemory
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_CudaIPC_checkGpuMemory
(JNIEnv *jenv, jclass jclz, jlong j_ptr, jint jcount)
{
  int *ptr = reinterpret_cast<int *>(j_ptr);
  std::vector<int> v(jcount);
  CUDF_CUDA_TRY(cudaMemcpy(v.data(), ptr, jcount * sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Checking GPU memory: ptr: " << j_ptr << " { ";
  for (int i = 0; i < jcount; i++) {
    std::cout << v[i] << " ";
  }
  std::cout << "}" << std::endl;
}


} // extern "C"
