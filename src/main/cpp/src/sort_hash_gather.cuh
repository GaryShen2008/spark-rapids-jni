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

#include <thrust/gather.h>

class SortHashGather {

public:
    // views of two tables to be joined
    // int first_bit: Likely used in the hash function.
    // int radix_bits: Used to determine the number of partitions.
    // int circular_buffer_size: Size of a circular buffer used in the join operation.
    explicit SortHashGather(cudf::table_view r_in, cudf::table_view s_in, cudf::column_view gather_map1, cudf::column_view gather_map2, int n_match, int circular_buffer_size, int first_bit,  int radix_bits)
    : r(r_in)
    , s(s_in)
    , g1(gather_map1)
    , g2(gather_map2)
    , n(n_match)
    , circular_buffer(circular_buffer_size)
    , first_bit(first_bit)
    , radix_bits(radix_bits)
    {
        nr = static_cast<int>(r.num_rows());
        ns = static_cast<int>(s.num_rows());

        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048));  // 1 Mc used, memory used now.
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048));  // 2 Mc used, memory used now.
        allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048)); // 3 Mc used, memory used now.
        allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048)); // 4 Mc used, memory used now.

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    std::unique_ptr<cudf::table> materialize_by_gather() {
        key_t* rkeys  {nullptr};
        key_t* skeys  {nullptr};

        key_t* rvals  {nullptr};
        key_t* svals  {nullptr};

        in_copy(&rkeys, r, 0);
        in_copy(&skeys, s, 0);
        auto r_cols = r.num_columns();
        auto s_cols = s.num_columns();

        // Assuming gather_map1 is your cudf::column_view
        int* r_match_idx; // Device pointer

        // Allocate memory for s_match_idx on the device
        cudaMalloc(&r_match_idx, g1.size() * sizeof(int));

        // Copy data from the column_view to the device pointer
        cudaMemcpy(r_match_idx, g1.data<int>(), g1.size() * sizeof(int), cudaMemcpyDeviceToDevice);

        // Assuming gather_map1 is your cudf::column_view
        int* s_match_idx; // Device pointer

        // Allocate memory for s_match_idx on the device
        cudaMalloc(&s_match_idx, g2.size() * sizeof(int));

        // Copy data from the column_view to the device pointer
        cudaMemcpy(s_match_idx, g2.data<int>(), g2.size() * sizeof(int), cudaMemcpyDeviceToDevice);



        // Create a cudf::table from the column
        std::vector<std::unique_ptr<cudf::column>> columns;

        for (int i = 1; i < r_cols; ++i) {

            std::cout << "I am inside gather" << std::endl;
            key_t* col {nullptr};
            cudaMalloc(&col, n * sizeof(key_t));

            in_copy(&rvals, r, i);
            if(i > 0) partition_pairs(rkeys, rvals, rkeys_partitions, (key_t*)rvals_partitions, nullptr, nr); // Mt + 2Mc is allocated.
            thrust::device_ptr<key_t> dev_data_ptr((key_t*)rvals_partitions);
            thrust::device_ptr<int> dev_idx_ptr(r_match_idx);

            thrust::device_ptr<key_t> dev_out_ptr(col);
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer, n), dev_data_ptr, dev_out_ptr);
            // First, create a column_view
            cudf::column_view col_view(
                cudf::data_type{cudf::type_to_id<key_t>()},
                static_cast<cudf::size_type>(n),
                col,  // assuming 'col' is a device pointer to your data
                nullptr,  // null mask (nullptr if no null values)
                0  // null count (0 if no null values)
            );

            print_column_view(col_view);

            // Then, create a cudf::column from the column_view
            auto col_column = std::make_unique<cudf::column>(col_view);

            std::cout << "I am after create a cudf::column from the column_view" << std::endl;

            std::cout << "Size of columns vector: " << columns.size() << std::endl;

            columns.push_back(std::move(col_column));
        }

        for (int i = 1; i < s_cols; ++i) {
            key_t* col {nullptr};
            cudaMalloc(&col, n * sizeof(key_t));

            in_copy(&svals, s, i);
            if(i > 0) partition_pairs(skeys, svals, skeys_partitions, (key_t*)svals_partitions, nullptr, ns);
            thrust::device_ptr<key_t> dev_data_ptr((key_t*)svals_partitions);
            thrust::device_ptr<int> dev_idx_ptr(s_match_idx);
            thrust::device_ptr<key_t> dev_out_ptr(col);
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer, n), dev_data_ptr, dev_out_ptr);
            // First, create a column_view
            cudf::column_view col_view(
                cudf::data_type{cudf::type_to_id<key_t>()},
                static_cast<cudf::size_type>(n),
                col,  // assuming 'col' is a device pointer to your data
                nullptr,  // null mask (nullptr if no null values)
                0  // null count (0 if no null values)
            );
            std::cout << "inside second columns gather" << columns.size() << std::endl;
            print_column_view(col_view);
            // Then, create a cudf::column from the column_view
            auto col_column = std::make_unique<cudf::column>(col_view);
            columns.push_back(std::move(col_column));
        }

        // Return the table
        return std::make_unique<cudf::table>(std::move(columns));

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

    ~SortHashGather() {

        release_mem(rkeys_partitions);
        release_mem(skeys_partitions);
        release_mem(rvals_partitions);
        release_mem(svals_partitions);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

public:
    float mat_time {0};
    float copy_device_vector_time{0};

private:

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


    template<typename KeyT, typename ValueT>
    void partition_pairs(KeyT*    keys,
                        ValueT*   values,
                        KeyT*     keys_out,
                        ValueT*   values_out,
                        int*      offsets,
                        const int num_items) {
        // offsets array to store offsets for each partition
        // num_items: number of key-value pairs to partition

        SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits);
        ssp.process();
    }

private:

    const cudf::table_view r;
    const cudf::table_view s;
    const cudf::column_view g1;
    const cudf::column_view g2;
    const int circular_buffer;

    using key_t = int32_t;

    int r_cols = r.num_columns();
    int s_cols = s.num_columns();

    int nr;
    int ns;
    int n;

    key_t* rkeys_partitions{nullptr};
    key_t* skeys_partitions{nullptr};
    void*  rvals_partitions{nullptr};
    void*  svals_partitions{nullptr};

    int first_bit;
    int radix_bits;

    cudaEvent_t start;
    cudaEvent_t stop;
};
