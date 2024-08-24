#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>

#include <iostream>
#include <memory>

#include "sort_hash_join.cuh"
#include "partitioned_hash_join.cuh"

using namespace cudf;
using size_type = cudf::size_type;

namespace spark_rapids_jni {

namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>, std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr){
    // Allocate device_uvector with 5 elements
    auto left_vector = std::make_unique<rmm::device_uvector<size_type>>(5, stream, mr);
    auto right_vector = std::make_unique<rmm::device_uvector<size_type>>(5, stream, mr);

    // Example: Fill vectors with dummy values (e.g., 0, 1, 2, 3, 4)
    std::vector<size_type> host_values = {0, 1, 2, 3, 4};
    cudaMemcpyAsync(left_vector->data(), host_values.data(), host_values.size() * sizeof(size_type), cudaMemcpyHostToDevice, stream.value());
    cudaMemcpyAsync(right_vector->data(), host_values.data(), host_values.size() * sizeof(size_type), cudaMemcpyHostToDevice, stream.value());
    // Return a pair of unique_ptrs to the vectors
    //SortHashJoin shj(left_input, right_input, 15, 0, 5);
    PartitionHashJoin phj(left_input, right_input, 6, 9, 0, 1000);
    phj.join();
    // shj.test_column_factories();
    return {std::move(left_vector), std::move(right_vector)};

}

} // detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
              std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr){
    return detail::inner_join(left_input, right_input, compare_nulls, stream, mr);
}
}
