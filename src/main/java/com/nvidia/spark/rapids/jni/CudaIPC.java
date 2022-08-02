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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.NativeDepsLoader;

public class CudaIPC {

  public static final int CUDA_IPC_HANDLE_SIZE = 64;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Returns the CUDA IPC memory handle for the given device pointer.
   *
   * @param devPtr the device pointer
   * @return the CUDA IPC memory handle
   */
  public static byte[] getCudaIpcMemHandle(long devPtr) {
    byte[] ret = getCudaIpcMemHandleBytes(devPtr);
    if (ret.length != CUDA_IPC_HANDLE_SIZE) {
      throw new RuntimeException("Unexpected CUDA IPC handle size: " + ret.length);
    }
    return ret;
  }

  private static native byte[] getCudaIpcMemHandleBytes(long devPtr);
  
  public static native void cudaDeviceSynchronize();

  public static native void checkGpuMemory(long ptr, int count);

}
