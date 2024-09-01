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

    int num_r = left_input.num_rows();
    int num_s = right_input.num_rows();
    int circular_buffer_size = std::max(num_r, num_s);
    // Return a pair of unique_ptrs to the vectors
    PartitionHashJoin phj(left_input, right_input, 6, 9, 0, circular_buffer_size);
    auto result = phj.join(stream, mr);
    std::cout << "partition_time " << phj.partition_time << std::endl;
    std::cout << "join_time " << phj.join_time << std::endl;
    std::cout << "copy_device_vector_time " << phj.copy_device_vector_time << std::endl;
    std::cout << "in_copy_time " << phj.in_copy_time << std::endl;
    //phj.print_match_indices();
    return result;
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
