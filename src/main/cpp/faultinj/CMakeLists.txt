#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

project(
  CUDA_FAULT_INJECTION
  VERSION 22.08.00
  LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)

add_library(
  cufaultinj SHARED
  faultinj.cu
)

find_path(SPDLOG_INCLUDE "spdlog"
    HINTS "$ENV{RMM_ROOT}/_deps/spdlog-src/include")

include_directories(
  "${SPDLOG_INCLUDE}"
)

target_link_libraries(
  cufaultinj CUDA::cupti_static
)
