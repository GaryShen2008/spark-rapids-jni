#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/filling.hpp>

#include <rmm/resource_ref.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "utils.cuh"

#include "phj_util.cuh"
#include "partition_util.cuh"

#include <cudf/column/column_view.hpp>

#include <iostream>

#include <cuda.h>
#include <cub/cub.cuh>

class SortHashJoinV1 {

public:
    explicit SortHashJoinV1(cudf::table_view const& r_in, cudf::table_view const& s_in, int first_bit,  int radix_bits, long circular_buffer_size, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    , stream(stream)
    , mr(mr)
    {
        nr = static_cast<int>(r.num_rows());
        ns = static_cast<int>(s.num_rows());

        //coarse_radix_bits = 6;

        n_partitions = (1 << radix_bits);
        //n_coarse_partitions = (1 << coarse_radix_bits);

        try {
            // Memory allocations with error handling
            d_n_matches = static_cast<unsigned long long *>(mr.allocate_async(sizeof(unsigned long long int), stream));
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_n_matches, 0, sizeof(unsigned long long int), stream));

            r_offsets = static_cast<int*>(mr.allocate_async(sizeof(int) * n_partitions, stream));
            s_offsets = static_cast<int*>(mr.allocate_async(sizeof(int) * n_partitions, stream));

            r_work = static_cast<uint64_t *>(mr.allocate_async(sizeof(uint64_t)* n_partitions * 2, stream));
            s_work = static_cast<uint64_t *>(mr.allocate_async(sizeof(uint64_t)* n_partitions * 2, stream));

            rkeys_partitions = static_cast<key_t *>(mr.allocate_async(sizeof(key_t)* (nr + 2048), stream));
            skeys_partitions = static_cast<key_t *>(mr.allocate_async(sizeof(key_t)* (ns + 2048), stream));

            rkeys_partitions_tmp = static_cast<key_t *>(mr.allocate_async(sizeof(key_t)* (nr + 2048), stream));
            skeys_partitions_tmp = static_cast<key_t *>(mr.allocate_async(sizeof(key_t)* (ns + 2048), stream));

            rvals_partitions = static_cast<int32_t *>(mr.allocate_async(sizeof(int32_t)* (nr + 2048), stream));
            svals_partitions = static_cast<int32_t *>(mr.allocate_async(sizeof(int32_t)* (ns + 2048), stream));
            total_work = static_cast<int *>(mr.allocate_async(sizeof(int), stream));
            CHECK_CUDA_ERROR(cudaMemsetAsync(total_work, 0, sizeof(int), stream));

            r_match_idx = static_cast<int *>(mr.allocate_async(sizeof(int) * circular_buffer_size, stream));
            s_match_idx = static_cast<int *>(mr.allocate_async(sizeof(int) * circular_buffer_size, stream));


        } catch (const std::exception& e) {
            // Handle any standard exceptions
            std::cerr << "n_partitions: " << n_partitions << "\n";
            std::cerr << "nr: " << nr << "\n";
            std::cerr << "ns: " << ns << "\n";
            std::cerr <<  "circular_buffer_size: " << circular_buffer_size << "\n";
            std::cerr << "Exception caught during memory allocation or kernel execution !!!:  " << e.what() << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        } catch (...) {
            // Catch all other exceptions
            std::cerr << "Unknown exception caught during memory allocation or kernel execution." << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        }


        // Kernel launches with error handling
        fill_sequence<<<num_tb(nr), 1024>>>((int*)(rkeys_partitions_tmp), 0, nr);
        //CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        fill_sequence<<<num_tb(ns), 1024>>>((int*)(skeys_partitions_tmp), 0, ns);


        //CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

    }

    static std::unique_ptr<cudf::table> gatherTest() {
        // Create a column with 5 integers
        auto column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), 5);
        auto mutable_view = column->mutable_view();

        // Fill the column with the value 1
        cudf::fill_in_place(mutable_view, 0, 5, cudf::numeric_scalar<int32_t>(1));

        // Create a table from the column
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.push_back(std::move(column));

        return std::make_unique<cudf::table>(std::move(columns));
    }

    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join(){

       TIME_FUNC_ACC(partition(), partition_time);
       join_copartitions();

       //std::cout << "n_matches: " << n_matches << "\n";
//        print_gpu_arr(r_match_idx, n_matches);
//        print_gpu_arr(s_match_idx, n_matches);

      //join_copartitions();

       try{
            //std::cout << "n_matches: " <<  n_matches << "\n";
            auto    r_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
            auto    s_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
            copy_device_vector(r_match_uvector, s_match_uvector, r_match_idx, s_match_idx);
            return std::make_pair(std::move(r_match_uvector), std::move(s_match_uvector));
       }
        catch (const std::exception& e) {
            // Handle any standard exceptions
            std::cerr << "Exception caught during partition or join kernel execution:3 " << e.what() << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        } catch (...) {
            // Catch all other exceptions
            std::cerr << "Unknown exception caught during partition or join kernel execution.3" << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        }
    }

    ~SortHashJoinV1() {
        mr.deallocate_async(d_n_matches, sizeof(unsigned long long int), stream);
        mr.deallocate_async(r_offsets, sizeof(int)* n_partitions, stream);
        mr.deallocate_async(s_offsets, sizeof(int)* n_partitions, stream);
        mr.deallocate_async(r_work, sizeof(uint64_t)* n_partitions * 2, stream);
        mr.deallocate_async(s_work, sizeof(uint64_t)* n_partitions*2, stream);
        mr.deallocate_async(rkeys_partitions, sizeof(key_t)*(nr+2048), stream);
        mr.deallocate_async(skeys_partitions, sizeof(key_t)*(ns+2048), stream);
        mr.deallocate_async(rvals_partitions, sizeof(key_t)*(nr+2048), stream);
        mr.deallocate_async(svals_partitions, sizeof(key_t)*(ns+2048), stream);
        mr.deallocate_async(total_work, sizeof(int), stream);
        mr.deallocate_async(r_match_idx, sizeof(int) * circular_buffer_size, stream);
        mr.deallocate_async(s_match_idx, sizeof(int) * circular_buffer_size, stream);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
//         release_mem(d_n_matches, sizeof(unsigned long long int), stream, mr);
//         release_mem(r_offsets, sizeof(int) * n_partitions, stream, mr);
//         release_mem(s_offsets, sizeof(int) * n_partitions, stream, mr);
//         release_mem(r_work, sizeof(uint64_t)*n_partitions*2, stream, mr);
//         release_mem(s_work, sizeof(uint64_t)*n_partitions*2, stream, mr);
//         release_mem(rkeys_partitions, sizeof(key_t)*(nr+2048), stream, mr);
//         release_mem(skeys_partitions, sizeof(key_t)*(ns+2048), stream, mr);
//         release_mem(rvals_partitions, sizeof(key_t)*(nr+2048), stream, mr);
//         release_mem(svals_partitions, sizeof(key_t)*(ns+2048), stream, mr);
//         release_mem(total_work, sizeof(int), stream, mr);
//
//         release_mem(r_match_idx, sizeof(int)*circular_buffer_size, stream, mr);
//         release_mem(s_match_idx, sizeof(int)*circular_buffer_size,stream, mr);

//         cudaEventDestroy(start);
//         cudaEventDestroy(stop);
    }

public:
    float partition_time {0};
    float partition_process_time1 {0};
    float partition_process_time2 {0};
    float join_time {0};
    float mat_time {0};
    float copy_device_vector_time{0};
    float partition_pair1 {0};
    float partition_pair2 {0};

private:

    void copy_device_vector(std::unique_ptr<rmm::device_uvector<cudf::size_type>> &r_match_uvector, std::unique_ptr<rmm::device_uvector<cudf::size_type>>& s_match_uvector,
    int*   r_match_idx , int* s_match_idx){
        try{
            if (r_match_uvector->data() == nullptr || r_match_idx == nullptr) {
                std::cerr << "Error: Null pointer detected" << std::endl;
            }

            if (n_matches < 0) {
                std::cerr << "Error: Invalid number of matches: " << n_matches << std::endl;
            }

            cudaError_t cudaStatus = cudaMemcpy(r_match_uvector->data(), r_match_idx,
                                                n_matches * sizeof(int), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                std::string errorMsg = "cudaMemcpy failed for r_match_idx.: ";
                errorMsg += cudaGetErrorString(cudaStatus);
                errorMsg += n_matches;
                std::cerr << "CUDA Error: " << errorMsg << std::endl;
                throw std::runtime_error(errorMsg);
            }

            cudaStatus = cudaMemcpy(s_match_uvector->data(), s_match_idx,
                                    n_matches * sizeof(int), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                std::string errorMsg = "cudaMemcpy failed for s_match_idx.: ";
                errorMsg += cudaGetErrorString(cudaStatus);
                std::cerr << "CUDA Error: " << errorMsg << std::endl;
                throw std::runtime_error(errorMsg);
            }

        }
        catch (const std::exception& e) {
            // Handle any standard exceptions
            std::cerr << "Exception caught during partition or join kernel execution:2 " << e.what() << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        } catch (...) {
            // Catch all other exceptions
            std::cerr << "Unknown exception caught during partition or join kernel execution.2" << std::endl;
            //cleanup_resources(); // Free any resources already allocated
            throw; // Re-throw the exception if necessary
        }

    }

    template<typename KeyT, typename ValueT>
    void partition_pairs(KeyT*    keys,
                        ValueT*   values,
                        KeyT*     keys_out,
                        ValueT*   values_out,
                        int*      offsets,
                        const int num_items) {
        // offsets array to store offsets for each partition
        SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits, stream, mr);
        ssp.process();
    }

//     void in_copy(key_t** arr, cudf::table_view table, int index){
//         cudf::column_view first_column = table.column(index);
//         cudf::data_type dtype_r = first_column.type();
//         const void* data_ptr_r;
//         if (dtype_r.id() == cudf::type_id::INT32) {
//             // The column type is INT32
//             data_ptr_r = static_cast<const void*>(first_column.data<int32_t>());
//             // Proceed with your INT32-specific logic here
//         } else {
//             // Handle other data types or throw an error if INT32 is required
//              throw std::runtime_error("R key type not supported");
//         }
//         *arr = const_cast<key_t*>(reinterpret_cast<const key_t*>(data_ptr_r));
//     }


    void partition() {

        key_t* rkeys  {nullptr};
        key_t* skeys  {nullptr};

        cudf::column_view key_column_r = r.column(0);
        cudf::data_type dtype_r = key_column_r.type();

        if(dtype_r.id() == cudf::type_id::INT32){
            rkeys = const_cast<key_t*>(key_column_r.data<int32_t>());
        }else {
            // Handle other data types or throw an error if INT32 is required
            throw std::runtime_error("R key type not supported");
        }

        cudf::column_view key_column_s = s.column(0);
        cudf::data_type dtype_s = key_column_s.type();

        if(dtype_s.id() == cudf::type_id::INT32){
            skeys = const_cast<key_t*>(key_column_s.data<int32_t>());
        }else {
            // Handle other data types or throw an error if INT32 is required
            throw std::runtime_error("R key type not supported");
        }

        partition_pairs(rkeys, (key_t*)nullptr, rkeys_partitions, (key_t*)rvals_partitions, r_coarse_offsets, nr);
        partition_pairs(skeys, (key_t*)nullptr, skeys_partitions, (key_t*)svals_partitions, s_coarse_offsets, ns);

        partition_pairs(rkeys, rkeys_partitions_tmp,
                                rkeys_partitions, (key_t*)rvals_partitions,
                                r_offsets, nr);

        partition_pairs(skeys, skeys_partitions_tmp,
                                skeys_partitions, (key_t*)svals_partitions,
                                s_offsets, ns);

        release_mem(rkeys_partitions_tmp, sizeof(key_t)*(nr+2048), stream, mr);
        release_mem(skeys_partitions_tmp, sizeof(key_t)*(ns+2048), stream, mr);
//         TIME_FUNC_ACC(partition_pairs(rkeys, rvals,
//                         rkeys_partitions, (key_t*)rvals_partitions,
//                         r_offsets, nr), partition_pair1);



        generate_work_units<<<num_tb(n_partitions,512),512>>>(r_offsets, s_offsets, r_work, s_work, total_work, n_partitions, threshold);
        //int total;
        //cudaMemcpy(&total, total_work, sizeof(total), cudaMemcpyDeviceToHost);
        //std::cout << "total work: " << total << "\n";
//         std::cout << "total work:";
//         print_gpu_arr(total_work, 1);
//         std::cout << "n partitions: " << n_partitions << "\n";
//
//         std::cout << "r_work 0 " ;
//         print_gpu_arr(r_work, 1);
//
//         std::cout << "s_work 0 ";
//         print_gpu_arr(s_work, 1);
//
//         std::cout << "r_work 1 " ;
//         print_gpu_arr(r_work + 1, 1);
//
//         std::cout << "s_work 1 ";
//         print_gpu_arr(s_work + 1, 1);
//
//         std::cout << "r_work 500 " ;
//         print_gpu_arr(r_work + 500, 1);
//
//         std::cout << "s_work 500 ";
//         print_gpu_arr(s_work + 500, 1);

        // Peek Mt + 4Mc
        // Used mem after exit = 4 Mc
        //print_gpu_arr(r_offsets, (size_t) n_partitions);
//         // for test identifying large partition
//         int* fine_partition_flags;
//         allocate_mem(&fine_partition_flags, false, sizeof(int) * n_partitions, stream, mr);
//         identify_large_partitions<<<(n_partitions + 255) / 256, 256>>>(r_offsets, n_partitions, 3, fine_partition_flags);
//         print_gpu_arr(fine_partition_flags, n_partitions);
//         int* prefix_sum;
//         allocate_mem(&prefix_sum, false, sizeof(int) * n_partitions, stream, mr);
//         fine_partition_prefix_sum(fine_partition_flags, prefix_sum, n_partitions, stream, mr);
//         print_gpu_arr(prefix_sum, n_partitions);

    }

    void join_copartitions() {
        // Allocate r_match_idx and s_match_idx(2Mc)
        // Peek mem = 6Mc
        CHECK_LAST_CUDA_ERROR();
        constexpr int NT = 512;
        constexpr int VT = 4;

        size_t sm_bytes = (bucket_size + 512) * (sizeof(key_t) + sizeof(int) + sizeof(int16_t)) + // elem, payload and next resp.
                       (1 << LOCAL_BUCKETS_BITS) * sizeof(int32_t) + // hash table head
                        + SHUFFLE_SIZE * (NT/32) * (sizeof(key_t) + sizeof(int)*2);
        std::cout << "sm_bytes: " << sm_bytes << std::endl;
//         cub::CountingInputIterator<int> r_itr(0);
//         cub::CountingInputIterator<int> s_itr(0);

        auto join_fn = join_copartitions_arr_v1<NT, VT, LOCAL_BUCKETS_BITS, SHUFFLE_SIZE, key_t, int, key_t>;
        cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
        join_fn<<<n_partitions, NT, sm_bytes>>>(
                                               rkeys_partitions, rvals_partitions,
                                               skeys_partitions, svals_partitions,
                                               r_work, s_work,
                                               total_work,
                                               radix_bits, 4096,
                                               d_n_matches,
                                               r_match_idx, s_match_idx,
                                                circular_buffer_size);

        CHECK_LAST_CUDA_ERROR();
        cudaMemcpy(&n_matches, d_n_matches, sizeof(n_matches), cudaMemcpyDeviceToHost);
    }


private:

    const cudf::table_view r;
    const cudf::table_view s;

    rmm::cuda_stream_view stream;
    rmm::device_async_resource_ref mr;

    using key_t = int32_t;

    int r_cols = r.num_columns();
    int s_cols = s.num_columns();

    static constexpr uint32_t log2_bucket_size = 12;
    static constexpr uint32_t bucket_size = (1 << log2_bucket_size);
    static constexpr int LOCAL_BUCKETS_BITS = 11;
    static constexpr int SHUFFLE_SIZE = sizeof(key_t) == 4 ? 32 : 16;
    int threshold = 2*bucket_size;

    int nr;
    int ns;
    unsigned long long int n_matches = 2;
    long circular_buffer_size;
    int first_bit;
    int n_partitions;
    int n_coarse_partitions;
    int radix_bits;
    int coarse_radix_bits = 6;

    unsigned long long int*   d_n_matches     {nullptr};
    int* r_coarse_offsets  {nullptr};
    int* s_coarse_offsets  {nullptr};
    int*   r_offsets       {nullptr};
    int*   s_offsets       {nullptr};
    uint64_t* r_work       {nullptr};
    uint64_t* s_work       {nullptr};
    int*   total_work      {nullptr};
    key_t* rkeys_partitions{nullptr};
    key_t* skeys_partitions{nullptr};
    key_t* rkeys_partitions_tmp{nullptr};
    key_t* skeys_partitions_tmp{nullptr};
    key_t*  rvals_partitions{nullptr};
    key_t*  svals_partitions{nullptr};
    int*   r_match_idx     {nullptr};
    int*   s_match_idx     {nullptr};

    cudaEvent_t start;
    cudaEvent_t stop;
};
