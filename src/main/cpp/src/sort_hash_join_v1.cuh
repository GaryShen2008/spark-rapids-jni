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
    explicit SortHashJoinV1(cudf::table_view r_in, cudf::table_view s_in, int first_bit,  int radix_bits, int circular_buffer_size, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
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

        coarse_radix_bits = 6;

        n_partitions = (1 << radix_bits);
        n_coarse_partitions = (1 << coarse_radix_bits);

        allocate_mem(&d_n_matches, true, sizeof(unsigned long long int), stream, mr);
        allocate_mem(&r_offsets, false, sizeof(int)*n_partitions, stream, mr);
        allocate_mem(&s_offsets, false, sizeof(int)*n_partitions, stream, mr);
        allocate_mem(&r_coarse_offsets, false, sizeof(int)*n_coarse_partitions, stream, mr);
        allocate_mem(&s_coarse_offsets, false, sizeof(int)*n_coarse_partitions, stream, mr);
        allocate_mem(&r_work,    false, sizeof(uint64_t)*n_partitions*2, stream, mr);
        allocate_mem(&s_work,    false, sizeof(uint64_t)*n_partitions*2, stream, mr);
        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048), stream, mr);  // 1 Mc used, memory used now.
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048), stream, mr);  // 2 Mc used, memory used now.
        allocate_mem(&rkeys_partitions_tmp, false, sizeof(key_t)*(nr+2048), stream, mr);  // 1 Mc used, memory used now.
        allocate_mem(&skeys_partitions_tmp, false, sizeof(key_t)*(ns+2048), stream, mr);  // 2 Mc used, memory used now.
        allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048), stream, mr); // 3 Mc used, memory used now.
        allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048), stream, mr); // 4 Mc used, memory used now.
        allocate_mem(&total_work, true, sizeof(int), stream, mr); // initialized to zero

        fill_sequence<<<num_tb(nr), 1024>>>((int*)(rkeys_partitions_tmp), 0, nr);
        fill_sequence<<<num_tb(ns), 1024>>>((int*)(skeys_partitions_tmp), 0, ns);

        allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size, stream, mr); // 5 Mc used
        allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size, stream, mr); // 6 Mc Used

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
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join(rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr){
        partition();
        join_copartitions();
        auto r_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
        auto s_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
        copy_device_vector(r_match_uvector, s_match_uvector, r_match_idx, s_match_idx);
        return std::make_pair(std::move(r_match_uvector), std::move(s_match_uvector));
    }

    ~SortHashJoinV1() {
        release_mem(d_n_matches, sizeof(unsigned long long int), stream, mr);
        release_mem(r_offsets, sizeof(int) * n_partitions, stream, mr);
        release_mem(s_offsets, sizeof(int) * n_partitions, stream, mr);
        release_mem(r_work, sizeof(uint64_t)*n_partitions*2, stream, mr);
        release_mem(s_work, sizeof(uint64_t)*n_partitions*2, stream, mr);
        release_mem(rkeys_partitions, sizeof(key_t)*(nr+2048), stream, mr);
        release_mem(skeys_partitions, sizeof(key_t)*(ns+2048), stream, mr);
        release_mem(rvals_partitions, sizeof(key_t)*(nr+2048), stream, mr);
        release_mem(svals_partitions, sizeof(key_t)*(ns+2048), stream, mr);
        release_mem(total_work, sizeof(int), stream, mr);

        release_mem(r_match_idx, sizeof(int)*circular_buffer_size, stream, mr);
        release_mem(s_match_idx, sizeof(int)*circular_buffer_size,stream, mr);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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

        if (r_match_uvector->data() == nullptr || r_match_idx == nullptr) {
            std::cerr << "Error: Null pointer detected" << std::endl;
        }

        if (n_matches < 0) {
            std::cerr << "Error: Invalid number of matches: " << n_matches << std::endl;
        }

        cudaError_t cudaStatus = cudaMemcpy(r_match_uvector->data(), r_match_idx,
                                            n_matches * sizeof(int), cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
            std::string errorMsg = "cudaMemcpy failed for r_match_idx: ";
            errorMsg += cudaGetErrorString(cudaStatus);
            std::cerr << "CUDA Error: " << errorMsg << std::endl;
            throw std::runtime_error(errorMsg);
        }

        cudaStatus = cudaMemcpy(s_match_uvector->data(), s_match_idx,
                                n_matches * sizeof(int), cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
            std::string errorMsg = "cudaMemcpy failed for r_match_idx: ";
            errorMsg += cudaGetErrorString(cudaStatus);
            std::cerr << "CUDA Error: " << errorMsg << std::endl;
            throw std::runtime_error(errorMsg);
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
        // num_items: number of key-value pairs to partition
        SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits, stream, mr);
        ssp.process();
    }

    void in_copy(key_t** arr, cudf::table_view table, int index){

        cudf::column_view first_column = table.column(index);
        cudf::data_type dtype_r = first_column.type();
        const void* data_ptr_r;
        if (dtype_r.id() == cudf::type_id::INT32) {
            // The column type is INT32
            data_ptr_r = static_cast<const void*>(first_column.data<int32_t>());
            // Proceed with your INT32-specific logic here
        } else {
            // Handle other data types or throw an error if INT32 is required
             throw std::runtime_error("R key type not supported");
        }
        *arr = const_cast<key_t*>(reinterpret_cast<const key_t*>(data_ptr_r));
    }


    void partition() {

        key_t* rkeys  {nullptr};
        key_t* skeys  {nullptr};

//         key_t* rvals  {nullptr};
//         key_t* svals  {nullptr};
        in_copy(&rkeys, r, 0);
        in_copy(&skeys, s, 0);
        //in_copy(&rvals, r, 0);
        //in_copy(&svals, s, 0);

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

        // Peek Mt + 2Mc
        //print_gpu_arr(r_offsets, (size_t) n_partitions);


        generate_work_units<<<num_tb(n_partitions,512),512>>>(s_offsets, r_offsets, s_work, r_work, total_work, n_partitions, threshold);
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
        //std::cout << "sm_bytes: " << sm_bytes << std::endl;
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
    int circular_buffer_size;
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
