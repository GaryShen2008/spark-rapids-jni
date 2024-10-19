#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <thrust/device_vector.h>
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
    explicit SortHashJoin(cudf::table_view r_in, cudf::table_view s_in, int first_bit,  int radix_bits, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    {
        nr = static_cast<int>(r.num_rows());
        ns = static_cast<int>(s.num_rows());

        // n_partitions is calculated as 2 raised to the power of radix_bits.
        n_partitions = (1 << radix_bits);

        //out.allocate(circular_buffer_size);

        allocate_mem(&d_n_matches);

        //using s_biggest_col_t = typename TupleS::biggest_col_t;
        //using r_biggest_col_t = typename TupleR::biggest_col_t;
        // here I assume every col is of type int32_t
        allocate_mem(&r_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&s_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&r_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&s_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048));  // 1 Mc used, memory used now.
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048));  // 2 Mc used, memory used now.
        allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048)); // 3 Mc used, memory used now.
        allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048)); // 4 Mc used, memory used now.
        allocate_mem(&total_work); // initialized to zero

        allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size); // 5 Mc used
        allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size); // 6 Mc Used

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void test_column_factories() {
        std::cout << "Hello I am here: " << std::endl;
        auto empty_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
        std::cout << "Empty column size: " << empty_col->size() << std::endl;

        auto numeric_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, 1000);
        std::cout << "Numeric column size: " << numeric_col->size() << std::endl;
    }

    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> join(rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr){
        std::cout << "hello I am 1." << std::endl;
        partition();
        std::cout << "hello I am 1.2." << std::endl;
        std::cout << "n_matches: " << n_matches << std::endl;
        join_copartitions();
        //TIME_FUNC_ACC(materialize_by_gather(), mat_time);
        std::cout << "n_matches: " << n_matches << std::endl;
        auto r_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);
        auto s_match_uvector = std::make_unique<rmm::device_uvector<cudf::size_type>>(n_matches, stream, mr);

        std::cout << "hello I am 1.5." << std::endl;

        TIME_FUNC_ACC(copy_device_vector(r_match_uvector, s_match_uvector,
            r_match_idx, s_match_idx), copy_device_vector_time);
        std::cout << "hello I am 2." << std::endl;
        // Return the pair of unique_ptrs to device_uvectors
        return std::make_pair(std::move(r_match_uvector), std::move(s_match_uvector));

    }

    ~SortHashJoin() {
        release_mem(d_n_matches);
        release_mem(r_offsets);
        release_mem(s_offsets);
        release_mem(rkeys_partitions);
        release_mem(skeys_partitions);
        release_mem(rvals_partitions);
        release_mem(svals_partitions);
        release_mem(r_work);
        release_mem(s_work);
        release_mem(total_work);

        release_mem(r_match_idx);
        release_mem(s_match_idx);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

public:
    float partition_time {0};
    float join_time {0};
    float mat_time {0};
    float copy_device_vector_time{0};

private:

    void copy_device_vector(std::unique_ptr<rmm::device_uvector<cudf::size_type>> &r_match_uvector, std::unique_ptr<rmm::device_uvector<cudf::size_type>>& s_match_uvector,
    int*   r_match_idx , int* s_match_idx){
        // Copy data from device to device_uvectors

        if (r_match_uvector->data() == nullptr || r_match_idx == nullptr) {
            std::cerr << "Error: Null pointer detected" << std::endl;
            // Handle error
        }

        if (n_matches <= 0) {
            std::cerr << "Error: Invalid number of matches: " << n_matches << std::endl;
            // Handle error
        }

        std::cout << "r_match_uvector->data(): " << r_match_uvector->data() << std::endl;
        std::cout << "r_match_idx: " << r_match_idx << std::endl;
        std::cout << "n_matches: " << n_matches << std::endl;
        std::cout << "Copy size: " << (n_matches * sizeof(int)) << " bytes" << std::endl;

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
        //
        SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits);
        ssp.process();
    }

    void in_copy(key_t** arr, cudf::table_view table, int index){

        // Get the column_view for the first column (index 0) because we only support single key join now.
        cudf::column_view first_column = table.column(index);
        std::cout << first_column.size() << std::endl;
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
        std::cout << "hello I am 1.1.1" << std::endl;
        in_copy(&rkeys, r, 0);
        in_copy(&skeys, s, 0);
        /*
        key_t* cpuRKeys = new key_t[nr];
        cudaMemcpy(cpuRKeys, rkeys, sizeof(key_t) * nr, cudaMemcpyDeviceToHost);

        if (cpuRKeys != nullptr) {
            for (int i = 0; i < nr; ++i) {
                std::cout << "cpuRKeys[" << i << "] = " << cpuRKeys[i] << std::endl;
            }
        } else {
            std::cout << "cpuRKeys is null" << std::endl;
        }
        */
        std::cout << "hello I am 1.1.2" << std::endl;
        partition_pairs(rkeys, rvals,
                        rkeys_partitions, (key_t*)rvals_partitions,
                        r_offsets, nr);
        std::cout << "hello I am 1.1.3" << std::endl;
        // Peek Mt + 2Mc
        partition_pairs(skeys, svals,
                        skeys_partitions, (key_t*)svals_partitions,
                        s_offsets, ns);

        std::cout << "hello I am 1.1.4" << std::endl;
        generate_work_units<<<num_tb(n_partitions,512),512>>>(r_offsets, s_offsets, r_work, s_work, total_work, n_partitions, threshold);
        // Peek Mt + 4Mc
        // Used mem after exit = 4 Mc
        std::cout << "hello I am 1.1.5" << std::endl;
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

    void materialize_by_gather() {
        // 2 already transformed payload columns
        // Alloc 0
        // Peek mem = 4Mc
        // Used after Exit 2Mc

        // Materialize a not yet transformed payload column
        // Mt + 2Mc allocated

        // after a column has been materialized Mt + Mc to be freed.transformed
        // Peek mem used = Mt + 4Mc

       // partition each payload columns and then gather
       /*
       for_<r_cols-1>([&](auto i) {
            using val_t = std::tuple_element_t<i.value+1, typename TupleR::value_type>;
            if(i.value > 0) partition_pairs(COL(r, 0), COL(r, i.value+1), rkeys_partitions, (val_t*)rvals_partitions, nullptr, nr); // Mt + 2Mc is allocated.
            thrust::device_ptr<val_t> dev_data_ptr((val_t*)rvals_partitions);
            thrust::device_ptr<int> dev_idx_ptr(r_match_idx);
            thrust::device_ptr<val_t> dev_out_ptr(COL(out, i.value+1));
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
       });
       for_<s_cols-1>([&](auto i) {
            constexpr auto k = i.value+r_cols;
            using val_t = std::tuple_element_t<i.value+1, typename TupleS::value_type>;
            if(i.value > 0) partition_pairs(COL(s, 0), COL(s, i.value+1), skeys_partitions, (val_t*)svals_partitions, nullptr, ns);
            thrust::device_ptr<val_t> dev_data_ptr((val_t*)svals_partitions);
            thrust::device_ptr<int> dev_idx_ptr(s_match_idx);
            thrust::device_ptr<val_t> dev_out_ptr(COL(out, k));
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
       });
       */
    }

private:

    const cudf::table_view r;
    const cudf::table_view s;

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
    int n_matches;
    int circular_buffer_size;
    int first_bit;
    int n_partitions;
    int radix_bits;

    int*   d_n_matches     {nullptr};
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
