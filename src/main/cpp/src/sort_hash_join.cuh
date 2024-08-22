#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cudf/table/table.hpp>

class SortHashJoin {
public:
    SortHashJoin(cudf::table_view r_in, cudf::table_view s_in, int first_bit, int radix_bits, int circular_buffer_size)
        : left_input(r_in), right_input(s_in), first_bit(first_bit), radix_bits(radix_bits), circular_buffer_size(circular_buffer_size) {
        // Initialize and allocate resources

    }

    // I guess no need because RMM can do this automatically.
    ~SortHashJoin() {
        // Release resources
    }

    std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
    std::unique_ptr<rmm::device_uvector<size_type>>> join() {
        // Implement the join logic
    }

    void print_stats() {
        // Print statistics
    }

private:
    // Member variables and methods
    void partition() {
        // Partition logic
    }

    void join_copartitions() {
        // Join copartitions logic
    }

    void materialize_by_gather() {
        // Materialize results logic
    }

    // Member variables
    table_view left_input,
    table_view right_input,
    int first_bit;
    int radix_bits;
    int circular_buffer_size;

    int nr;
    int ns;
    int n_matches;
    int n_partitions;
    int radix_bits;
};