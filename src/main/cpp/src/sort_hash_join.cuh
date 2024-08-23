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

#include <iostream>

#include <cuda.h>
#include <cub/cub.cuh>

class SortHashJoin {

public:
    explicit SortHashJoin(cudf::table_view r_in, cudf::table_view s_in, int first_bit,  int radix_bits, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    {
        nr = static_cast<int>(r.num_rows());
        ns = static_cast<int>(s.num_rows());
        n_partitions = (1 << radix_bits);

        allocate_mem(&d_n_matches);

        allocate_mem(&r_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&s_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&r_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&s_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048));
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048));
        allocate_mem(&total_work); // initialized to zero

        // late materialization
        allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size);
        allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

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

    void test_column_factories() {
        std::cout << "Hello I am here: " << std::endl;
        auto empty_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
        std::cout << "Empty column size: " << empty_col->size() << std::endl;

        auto numeric_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, 1000);
        std::cout << "Numeric column size: " << numeric_col->size() << std::endl;
    }

    void join(){


    }

    ~SortHashJoin() {}

private:

    void partition() {

    }

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

    int*   d_n_matches     {nullptr};
    int*   r_offsets       {nullptr};
    int*   s_offsets       {nullptr};
    uint64_t* r_work       {nullptr};
    uint64_t* s_work       {nullptr};
    int*   total_work      {nullptr};
    key_t* rkeys_partitions{nullptr};
    key_t* skeys_partitions{nullptr};

    int* r_match_idx {};
    int* s_match_idx {};

    cudaEvent_t start;
    cudaEvent_t stop;
};