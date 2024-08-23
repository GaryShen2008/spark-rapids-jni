#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <iostream>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

class SortHashJoin {

public:
    explicit SortHashJoin(cudf::table_view& r_in, cudf::table_view& s_in, int first_bit,  int radix_bits, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    {
        test_mem();
        // first_bit, radix bits, int circular_buffer_size: Parameters for partitioning and buffer size.
        // nr and ns store the number of items in r and s
        // n_partitions calculated the number of partitions based on radix_bits.

        // Use t
        // Allocate Output Buffer:
        // Allocate Memory for Intermediate Data Structures:
        // Partition offsets(r_offsets, s_offsets)
        // Work arrays (r_work, s_work)
        // Key and value partitions (rkeys_partitions, skeys_partitions, rvals_partitions, svals_partitions).
        // Total work counter (total_work).
        // Match indices (r_match_idx, s_match_idx) if late materialization is used.
        // Create CUDA Events for measuring execution time.
    }

    // Function to print the column data
    void print_column(const cudf::column_view& col) {
        // Transfer data from device to host
        std::vector<int> host_data(col.size());
        cudaMemcpy(host_data.data(), col.data<int>(), col.size() * sizeof(int), cudaMemcpyDeviceToHost);

        // Print data
        for (int i = 0; i < col.size(); ++i) {
            std::cout << host_data[i] << " ";
        }
        std::cout << std::endl;
    }

    void test_mem(){

            // Number of partitions
            int n_partitions = 10;

            // Create a numeric column of INT32 with n_partitions elements
            auto r_offsets = make_numeric_column(cudf::data_type{cudf::type_id::INT32}, n_partitions);

            // Fill the column with some data
            {
                // Get a mutable view of the column
                auto r_offsets_view = r_offsets->mutable_view();

                // Create a device vector to hold data
                rmm::device_uvector<int> d_data(n_partitions, rmm::cuda_stream_default);

                // Fill the device vector with incremental values
                thrust::sequence(thrust::device, d_data.begin(), d_data.end(), 0);

                // Copy data from device vector to column
                cudaMemcpy(r_offsets_view.data<int>(), d_data.data(), n_partitions * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            // Print the column
            print_column(r_offsets->view());
    }


    ~SortHashJoin() {}

private:
    void partition_pairs() {}

    const cudf::table_view r;
    const cudf::table_view s;

    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;
    int first_bit;
    int n_partitions;
    int radix_bits;

    int* r_match_idx {};
    int* s_match_idx {};
};