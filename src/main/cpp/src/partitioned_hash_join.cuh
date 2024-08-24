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

#include <cudf/column/column_view.hpp>

#include <iostream>

#include <cuda.h>
#include <cub/cub.cuh>

class PartitionHashJoin {

public:
    explicit PartitionHashJoin(cudf::table_view r_in, cudf::table_view s_in, int log_parts1, int log_parts2, int first_bit, int circular_buffer_size)
    : r(r_in)
    , s(s_in)
    , log_parts1(log_parts1)
    , log_parts2(log_parts2)
    , first_bit(first_bit)
    , circular_buffer_size(circular_buffer_size)
    {
        nr = static_cast<int>(r.num_rows());
        ns = static_cast<int>(s.num_rows());

        parts1 = 1 << log_parts1;
        parts2 = 1 << (log_parts1 + log_parts2);

        buckets_num_max_R    = ((((nr + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
        // we consider two extreme cases here
        // (1) S keys are uniformly distributed across partitions; (2) S keys concentrate in one partition
        buckets_num_max_S    = std::max(((((ns + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2,
                                        (ns+bucket_size-1)/bucket_size+parts2);

        allocate_mem(&r_key_partitions, true, buckets_num_max_R * bucket_size * sizeof(key_t));
        allocate_mem(&s_key_partitions, true, buckets_num_max_S * bucket_size * sizeof(key_t));

        // Get the column_view for the first column (index 0)
        cudf::column_view first_column = table[0];

        cudf::data_type dtype = first_column.type();
        cudf::type_id type_id = dtype.id();
        void* data_ptr = nullptr;

        switch(type_id) {
               case cudf::type_id::INT32:
                    data_ptr = const_cast<void*>(static_cast<const void*>(column_view.data<int32_t>()));
                    break;
               case cudf::type_id::FLOAT64:
                    data_ptr = const_cast<void*>(static_cast<const void*>(column_view.data<double>()));
                    break;
               case cudf::type_id::STRING:
                    data_ptr = const_cast<void*>(static_cast<const void*>(column_view.data<cudf::string_view>()));
                    break;
               // ... handle other types as needed
               default:
                    // Handle unexpected types
                    throw std::runtime_error("Unsupported data type");

        }
        //cudaMemcpy(r_key_partitions, COL(r,0), nr*sizeof(key_t), cudaMemcpyDefault);
        //cudaMemcpy(s_key_partitions, COL(s,0), ns*sizeof(key_t), cudaMemcpyDefault);
//  #ifndef CHECK_CORRECTNESS
//          release_mem(COL(r,0));
//          release_mem(COL(s,0));
//  #endif
        allocate_mem(&r_key_partitions_temp, true, buckets_num_max_R * bucket_size * sizeof(key_t));
        allocate_mem(&s_key_partitions_temp, true, buckets_num_max_S * bucket_size * sizeof(key_t));


        // late materialization
        allocate_mem(&r_val_partitions, true, buckets_num_max_R * bucket_size * sizeof(int32_t));
        allocate_mem(&r_val_partitions_temp, true, buckets_num_max_R * bucket_size * sizeof(int32_t));
        allocate_mem(&s_val_partitions, true, buckets_num_max_S * bucket_size * sizeof(int32_t));
        allocate_mem(&s_val_partitions_temp, true, buckets_num_max_S * bucket_size * sizeof(int32_t));
        allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size);
        allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size);
        fill_sequence<<<num_tb(nr), 1024>>>((int*)(r_val_partitions), 0, nr);
        fill_sequence<<<num_tb(ns), 1024>>>((int*)(s_val_partitions), 0, ns);

        for (int i = 0; i < 2; i++) {
            allocate_mem(&chains_R[i], false, buckets_num_max_R * sizeof(uint32_t));
            allocate_mem(&cnts_R[i], false, parts2 * sizeof(uint32_t));
            allocate_mem(&heads_R[i], false, parts2 * sizeof(uint64_t));
            allocate_mem(&buckets_used_R[i], false, sizeof(uint32_t));

            allocate_mem(&chains_S[i], false, buckets_num_max_S * sizeof(uint32_t));
            allocate_mem(&cnts_S[i], false, parts2 * sizeof(uint32_t));
            allocate_mem(&heads_S[i], false, parts2 * sizeof(uint64_t));
            allocate_mem(&buckets_used_S[i], false, sizeof(uint32_t));
        }

        bucket_info_R = (uint32_t*)s_val_partitions_temp;

        allocate_mem(&d_n_matches);

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

    ~PartitionHashJoin() {}

private:

    void partition() {

    }

    void partition_pairs() {}


    static constexpr uint32_t log2_bucket_size = 12;
    static constexpr uint32_t bucket_size = (1 << log2_bucket_size);

    const cudf::table_view r;
    const cudf::table_view s;

    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;

    int first_bit;
    int parts1;
    int parts2;
    int log_parts1;
    int log_parts2;
    size_t buckets_num_max_R;
    size_t buckets_num_max_S;

    key_t* r_key_partitions      {nullptr};
    key_t* s_key_partitions      {nullptr};
    key_t* r_key_partitions_temp {nullptr};
    key_t* s_key_partitions_temp {nullptr};

    void* r_val_partitions       {nullptr};
    void* s_val_partitions       {nullptr};
    void* r_val_partitions_temp  {nullptr};
    void* s_val_partitions_temp  {nullptr};

    // meta info for bucket chaining
    uint32_t* chains_R[2];
    uint32_t* chains_S[2];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[2];

    uint64_t* heads_R[2];
    uint64_t* heads_S[2];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[2];

    uint32_t* bucket_info_R{nullptr};

    int*   r_match_idx     {nullptr};
    int*   s_match_idx     {nullptr};

    int*   d_n_matches     {nullptr};
};