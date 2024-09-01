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
#include <iostream>
#include <vector>
#include "mem_manager.hpp" // Assume this is the file containing the UserSpaceMM class

using namespace cudf;

struct Mem_ManagerTests : public test::BaseFixture {};

TEST_F(Mem_ManagerTests, test)
{
    constexpr size_t POOL_SIZE = 1024 * 1024;  // 1 MB pool
    UserSpaceMM<POOL_SIZE> mm;

    std::vector<void*> allocations;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    try {
        // Allocate various sizes
        for (int i = 0; i < 10; ++i) {
            void* ptr;
            size_t size = (i + 1) * 1024;  // Increasing sizes
            mm.allocate(&ptr, true, size, stream);
            allocations.push_back(ptr);
            std::cout << "Allocated " << size << " bytes at " << ptr << std::endl;
        }

        // Free every other allocation
        for (size_t i = 0; i < allocations.size(); i += 2) {
            mm.release(allocations[i], stream);
            std::cout << "Freed allocation at " << allocations[i] << std::endl;
        }

        // Try to allocate a large block
        void* large_ptr;
        mm.allocate(&large_ptr, false, POOL_SIZE / 2, stream);
        std::cout << "Allocated large block of " << POOL_SIZE / 2 << " bytes at " << large_ptr << std::endl;

        // Free remaining allocations
        for (size_t i = 1; i < allocations.size(); i += 2) {
            mm.release(allocations[i], stream);
        }
        mm.release(large_ptr, stream);

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Peak memory usage: " << mm.get_peak_mem_used() << " bytes" << std::endl;

    cudaStreamDestroy(stream);

}
