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
    explicit SortHashJoin(cudf::table_view& r_in, cudf::table_view& s_in, int first_bit,  int radix_bits, int circular_buffer_size)
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
        // Test make_empty_column
        auto empty_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
        std::cout << "Empty column size: " << empty_col->size() << std::endl;

        // Test make_numeric_column
        auto numeric_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, 1000);
        std::cout << "Numeric column size: " << numeric_col->size() << std::endl;

        // Test make_strings_column
        thrust::device_vector<const char*> d_strings = {"hello", "world", "cudf"};
        thrust::device_vector<size_t> d_string_lengths = {5, 5, 4};
        auto string_col = cudf::make_strings_column(
            cudf::device_span<thrust::pair<const char*, size_t> const>(
                thrust::make_zip_iterator(d_strings.begin(), d_string_lengths.begin()),
                thrust::make_zip_iterator(d_strings.end(), d_string_lengths.end())
            )
        );
        cudf::strings_column_view scv(string_col->view());
        std::cout << "Strings column size: " << scv.size() << std::endl;

        // Test make_column_from_scalar
        cudf::numeric_scalar<int32_t> scalar(42);
        auto scalar_col = cudf::make_column_from_scalar(scalar, 100);
        std::cout << "Scalar column size: " << scalar_col->size() << std::endl;

        // Test make_dictionary_column
        auto keys = cudf::make_strings_column({"a", "b", "c", "d"});
        rmm::device_vector<int32_t> indices{1, 0, 2, 3, 1, 2};
        auto indices_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, indices.size());
        cudaMemcpy(indices_col->mutable_view().data<int32_t>(), indices.data().get(),
                   indices.size() * sizeof(int32_t), cudaMemcpyDeviceToDevice);

        auto dict_col = cudf::make_dictionary_column(std::move(keys), std::move(indices_col));
        cudf::dictionary_column_view dcv(dict_col->view());
        std::cout << "Dictionary column size: " << dcv.size() << std::endl;
        std::cout << "Dictionary keys size: " << dcv.keys().size() << std::endl;
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