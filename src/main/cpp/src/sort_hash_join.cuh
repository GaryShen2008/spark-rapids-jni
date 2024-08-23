#pragma once
#include <cudf/table/table_view.hpp>

class SortHashJoin {

public:
    explicit SortHashJoin(cudf::table_view& r_in, cudf::table_view& s_in, int first_bit,  int radix_bits, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    , radix_bits(radix_bits)
    {

    }
`                                                                   `   +
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
};