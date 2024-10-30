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

#include <cudf_test/base_fixture.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <iostream>
#include <vector>
#include <rmm/resource_ref.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace cudf;

void alloc_by_cuda(void** ptr, bool clear, size_t sz, rmm::device_async_resource_ref mr, rmm::cuda_stream_view stream){
    *ptr = mr.allocate_async(sz, stream);
    //CHECK_CUDA_ERROR(cudaMallocAsync(ptr, sz, stream));
    if(clear) cudaMemsetAsync(*ptr, 0, sz, stream);
}

template<typename T>
void release_mem(T* ptr, size_t sz, rmm::device_async_resource_ref mr, rmm::cuda_stream_view stream) {
    mr.deallocate_async(ptr, sz, stream);
}

template<typename T>
void print_gpu_arr(const T* arr, size_t n, size_t offset=0) {
    T* temp = new T[n];
    cudaMemcpy(temp, arr+offset, sizeof(T)*n, cudaMemcpyDeviceToHost);
    printf("%p: ", arr);
    for(auto i = 0; i < (int)n; i++) {
        std::cout << temp[i] << " ";
    }
    std::cout << std::endl;
    delete [] temp;
}


struct Mem_ManagerTests : public test::BaseFixture {};
//
//TEST_F(Mem_ManagerTests, test)
//{
//    constexpr size_t POOL_SIZE = 1024 * 1024;  // 1 MB pool
//    UserSpaceMM<POOL_SIZE> mm;
//
//    std::vector<void*> allocations;
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    try {
//        // Allocate various sizes
//        for (int i = 0; i < 10; ++i) {
//            void* ptr;
//            size_t size = (i + 1) * 1024;  // Increasing sizes
//            mm.allocate(&ptr, true, size, stream);
//            allocations.push_back(ptr);
//            std::cout << "Allocated " << size << " bytes at " << ptr << std::endl;
//        }
//
//        // Free every other allocation
//        for (size_t i = 0; i < allocations.size(); i += 2) {
//            mm.release(allocations[i], stream);
//            std::cout << "Freed allocation at " << allocations[i] << std::endl;
//        }
//
//        // Try to allocate a large block
//        void* large_ptr;
//        mm.allocate(&large_ptr, false, POOL_SIZE / 2, stream);
//        std::cout << "Allocated large block of " << POOL_SIZE / 2 << " bytes at " << large_ptr << std::endl;
//
//        // Free remaining allocations
//        for (size_t i = 1; i < allocations.size(); i += 2) {
//            mm.release(allocations[i], stream);
//        }
//        mm.release(large_ptr, stream);
//
//    } catch (const std::exception& e) {
//        std::cerr << "Exception caught: " << e.what() << std::endl;
//    }
//
//    std::cout << "Peak memory usage: " << mm.get_peak_mem_used() << " bytes" << std::endl;
//
//    cudaStreamDestroy(stream);
//
//}

TEST_F(Mem_ManagerTests, test2)
{
    rmm::cuda_stream_view stream      = cudf::get_default_stream();
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    int* test {nullptr};
    size_t n = 20;
    alloc_by_cuda((void**)&test, false, n * sizeof(int), mr, stream);
    print_gpu_arr(test, n);
    release_mem(test, n * sizeof(int), mr , stream);
    print_gpu_arr(test, n);
}