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

class SortHashJoin {

public:
    // views of two tables to be joined
    // int first_bit: Likely used in the hash function.
    // int radix_bits: Used to determine the number of partitions.
    // int circular_buffer_size: Size of a circular buffer used in the join operation.
    explicit SortHashJoin(cudf::table_view r_in, cudf::table_view s_in, int first_bit,  int radix_bits, int circular_buffer_size, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
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

        // n_partitions is calculated as 2 raised to the power of radix_bits.
        n_partitions = (1 << radix_bits);

        //out.allocate(circular_buffer_size);

        allocate_mem(&d_n_matches, true, sizeof(unsigned long long int), stream, mr);

        allocate_mem(&r_offsets, false, sizeof(int)*n_partitions, stream, mr);
        allocate_mem(&s_offsets, false, sizeof(int)*n_partitions, stream, mr);
        allocate_mem(&r_work,    false, sizeof(uint64_t)*n_partitions*2, stream, mr);
        allocate_mem(&s_work,    false, sizeof(uint64_t)*n_partitions*2, stream, mr);
        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048), stream, mr);  // 1 Mc used, memory used now.
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048), stream, mr);  // 2 Mc used, memory used now.
        allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048), stream, mr); // 3 Mc used, memory used now.
        allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048), stream, mr); // 4 Mc used, memory used now.
        allocate_mem(&total_work, true, sizeof(int), stream, mr); // initialized to zero

        allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size, stream, mr); // 5 Mc used
        allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size, stream, mr); // 6 Mc Used

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void test_column_factories() {
        auto empty_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
        auto numeric_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, 1000);
    }

    void print_column_view(cudf::column_view const& col) {
        // Check the type of the column
        if (col.type().id() == cudf::type_id::INT32) {
            // Create a device vector from the column data
            thrust::device_vector<int> d_data(col.begin<int>(), col.end<int>());

            // Copy to host
            thrust::host_vector<int> h_data = d_data;

            // Print the data
            for (auto const& val : h_data) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        else {
            // Handle other types as needed
            std::cout << "Unsupported type" << std::endl;
        }
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
        //TIME_FUNC_ACC(join_copartitions(), join_time);
        join_copartitions();
        //TIME_FUNC_ACC(materialize_by_gather(), mat_time);
        //std::cout << "n_matches: " << n_matches << std::endl;
        auto r_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
        auto s_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);

        //TIME_FUNC_ACC(copy_device_vector(r_match_uvector, s_match_uvector,
            //r_match_idx, s_match_idx), copy_device_vector_time);
        copy_device_vector(r_match_uvector, s_match_uvector,
                        r_match_idx, s_match_idx);

        // Return the pair of unique_ptrs to device_uvectors
        return std::make_pair(std::move(r_match_uvector), std::move(s_match_uvector));

    }

    ~SortHashJoin() {
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
        // Copy data from device to device_uvectors

        if (r_match_uvector->data() == nullptr || r_match_idx == nullptr) {
            std::cerr << "Error: Null pointer detected" << std::endl;
            // Handle error
        }

        if (n_matches < 0) {
            std::cerr << "Error: Invalid number of matches: " << n_matches << std::endl;
            // Handle error
        }

//         std::cout << "r_match_uvector->data(): " << r_match_uvector->data() << std::endl;
//         std::cout << "r_match_idx: " << r_match_idx << std::endl;
//         std::cout << "n_matches: " << n_matches << std::endl;
//         std::cout << "Copy size: " << (n_matches * sizeof(int)) << " bytes" << std::endl;

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
//         if(partition_process_time1 == 0){
//             TIME_FUNC_ACC(ssp.process(), partition_process_time1);
//         }
//         else{
//             TIME_FUNC_ACC(ssp.process(), partition_process_time2);
//         }
        ssp.process();
    }

    void in_copy(key_t** arr, cudf::table_view table, int index){

        // Get the column_view for the first column (index 0) because we only support single key join now.
        cudf::column_view first_column = table.column(index);
        //std::cout << first_column.size() << std::endl;
        // Get the type of the first column.
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

        key_t* rvals  {nullptr};
        key_t* svals  {nullptr};
        in_copy(&rkeys, r, 0);
        in_copy(&skeys, s, 0);
        in_copy(&rvals, r, 0);
        in_copy(&svals, s, 0);

//         TIME_FUNC_ACC(partition_pairs(skeys, svals,
//                         skeys_partitions, (key_t*)svals_partitions,
//                         s_offsets, ns), partition_pair2);
        partition_pairs(skeys, svals,
                                skeys_partitions, (key_t*)svals_partitions,
                                s_offsets, ns);

//         TIME_FUNC_ACC(partition_pairs(rkeys, rvals,
//                         rkeys_partitions, (key_t*)rvals_partitions,
//                         r_offsets, nr), partition_pair1);
        partition_pairs(rkeys, rvals,
                                rkeys_partitions, (key_t*)rvals_partitions,
                                r_offsets, nr);
        // Peek Mt + 2Mc



        generate_work_units<<<num_tb(n_partitions,512),512>>>(r_offsets, s_offsets, r_work, s_work, total_work, n_partitions, threshold);
        // Peek Mt + 4Mc
        // Used mem after exit = 4 Mc

//         key_t* h_rkeys_partitions = new key_t[nr];;
//         cudaMemcpy(h_rkeys_partitions, rkeys_partitions, sizeof(key_t)*nr, cudaMemcpyDeviceToHost);
//         for (long i = 0; i < nr; ++i) {
//             std::cout << h_rkeys_partitions[i] << " ";
//         }
//
//         std::cout << std::endl;
//         delete[] h_rkeys_partitions;

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
        cub::CountingInputIterator<int> r_itr(0);
        cub::CountingInputIterator<int> s_itr(0);

        auto join_fn = join_copartitions_arr<NT, VT, LOCAL_BUCKETS_BITS, SHUFFLE_SIZE, key_t, int>;
        cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
        join_fn<<<n_partitions, NT, sm_bytes>>>(
                                               rkeys_partitions, r_itr,
                                               skeys_partitions, s_itr,
                                               r_work, s_work,
                                               total_work,
                                               radix_bits, 4096,
                                               d_n_matches,
                                               r_match_idx, s_match_idx,
                                                circular_buffer_size);

        CHECK_LAST_CUDA_ERROR();
        cudaMemcpy(&n_matches, d_n_matches, sizeof(n_matches), cudaMemcpyDeviceToHost);
        // free 2Mc
        // Used mem after exit = 4Mc
    }


private:

    const cudf::table_view r;
    const cudf::table_view s;

    rmm::cuda_stream_view stream;
    rmm::device_async_resource_ref mr;

    using key_t = int32_t;

    int r_cols = r.num_columns();
    int s_cols = s.num_columns();
    bool kAlwaysLateMaterialization = false;
    bool r_materialize_early = (r_cols == 2 && !kAlwaysLateMaterialization);
    bool s_materialize_early = (s_cols == 2 && !kAlwaysLateMaterialization);
    bool early_materialization = (r_materialize_early && s_materialize_early);
    static constexpr uint32_t log2_bucket_size = 12;
    static constexpr uint32_t bucket_size = (1 << log2_bucket_size);
    static constexpr int LOCAL_BUCKETS_BITS = 11;
    static constexpr int SHUFFLE_SIZE = sizeof(key_t) == 4 ? 32 : 16;
    int threshold = 2*bucket_size;

    //using key_t = std::tuple_element_t<0, typename TupleR::value_type>;
    //using key_t = std::tuple_element_t<1, typename TupleR::value_type>;
    //using key_t = std::tuple_element_t<1, typename TupleS::value_type>;
    //cudf::table_view out;

    int nr;
    int ns;
    unsigned long long int n_matches;
    int circular_buffer_size;
    int first_bit;
    int n_partitions;
    int radix_bits;

    unsigned long long int*   d_n_matches     {nullptr};
    int*   r_offsets       {nullptr};
    int*   s_offsets       {nullptr};
    uint64_t* r_work       {nullptr};
    uint64_t* s_work       {nullptr};
    int*   total_work      {nullptr};
    key_t* rkeys_partitions{nullptr};
    key_t* skeys_partitions{nullptr};
    void*  rvals_partitions{nullptr};
    void*  svals_partitions{nullptr};
    int*   r_match_idx     {nullptr};
    int*   s_match_idx     {nullptr};

    cudaEvent_t start;
    cudaEvent_t stop;
};
