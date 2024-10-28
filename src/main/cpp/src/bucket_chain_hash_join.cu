#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <iostream>
#include <memory>

#include "sort_hash_join.cuh"
#include "sort_hash_gather.cuh"
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
    SortHashJoin shj(left_input, right_input, 0, 15, circular_buffer_size);
    auto result = shj.join(stream, mr);
    return result;
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
    SortHashGather shg(source_table, gather_map, n_match, num_rows, 0, 15);
    auto result = shg.materialize_by_gather();;
    return result;

}

} // detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
              std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls = null_equality::EQUAL,
           rmm::cuda_stream_view stream = cudf::get_default_stream(),
           rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()){
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
