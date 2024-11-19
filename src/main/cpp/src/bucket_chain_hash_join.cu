#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/types.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/join.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <iostream>
#include <memory>

#include "sort_hash_join_v1.cuh"
#include "sort_hash_gather.cuh"
#include "partitioned_hash_join.cuh"
#include "bucket_chain_hash_join.hpp"

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
  // Following code is copied from cudf join.cu
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},
    stream,
    cudf::get_current_device_resource_ref());  // temporary objects returned

  // now rebuild the table views with the updated ones
  auto const left      = matched.second.front();
  auto const right     = matched.second.back();
  auto const has_nulls = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                           ? cudf::nullable_join::YES
                           : cudf::nullable_join::NO;

  int num_r = left.num_rows();
  int num_s = right.num_rows();
  // circular_buffer_size and the radix bits should not set hardcode like this.
  int circular_buffer_size = std::max(num_r, num_s);
  if(num_s > num_r){
    SortHashJoinV1 shj(left, right, 0, 17, circular_buffer_size, stream, mr);
    return shj.join(stream, mr);
  } else {
    SortHashJoinV1 shj(right, left, 0, 17, circular_buffer_size, stream, mr);
    auto [right_result, left_result] = shj.join(stream, mr);
    return std::pair(std::move(left_result), std::move(right_result));
  }
  //std::cout << "partition time: " << shj.partition_time << std::endl;
  //std::cout << "join time: "<< shj.join_time << std::endl;
  //     std::cout << "copy_device_vector_time: "<< shj.copy_device_vector_time << std::endl;
  //     std::cout << "partition_pair1 time: "<< shj.partition_pair1 << std::endl;
  //     std::cout << "partition_pair2 time: "<< shj.partition_pair2 << std::endl;
  //std::cout << "partition_process1 time: "<< shj.partition_process_time1 << std::endl;
  //std::cout << "partition_process2 time: "<< shj.partition_process_time2 << std::endl;
}

std::unique_ptr<table> gather(cudf::table_view const& source_table,
                              cudf::column_view const& gather_map,
                              cudf::out_of_bounds_policy bounds_policy,
                              cudf::detail::negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
    int n_match = gather_map.size();
    int num_rows = source_table.num_rows();
    SortHashGather shg(source_table, gather_map, n_match, num_rows, 0, 17, stream, mr);
    auto result = shg.materialize_by_gather();;
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

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  //CUDF_FUNC_RANGE();

  auto index_policy = is_unsigned(gather_map.type()) ? cudf::detail::negative_index_policy::NOT_ALLOWED
                                                     : cudf::detail::negative_index_policy::ALLOWED;

  return detail::gather(source_table, gather_map, bounds_policy, index_policy, stream, mr);
}

}
