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
#include <iostream>

class SortHashJoin {

public:
    explicit SortHashJoin(cudf::table_view r_in, cudf::table_view s_in, int first_bit,  int radix_bits, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    {
        void test_column_factories();
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
       std::cout << "Hello I am here: " std::endl;
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