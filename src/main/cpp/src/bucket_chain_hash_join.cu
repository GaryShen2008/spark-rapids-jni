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
    // Return a pair of unique_ptrs to the vectors
    //PartitionHashJoin phj(left_input, right_input, 6, 9, 0, circular_buffer_size);
    //auto result = phj.join(stream, mr);
    SortHashJoin shj(left_input, right_input, 0, 15, circular_buffer_size);
    auto result = shj.join(stream, mr);
    //std::cout << "partition_time " << phj.partition_time << std::endl;
    //std::cout << "join_time " << phj.join_time << std::endl;
    //std::cout << "copy_device_vector_time " << phj.copy_device_vector_time << std::endl;
    //std::cout << "in_copy_time " << phj.in_copy_time << std::endl;
    //phj.print_match_indices();
    return result;
}

std::unique_ptr<table> gather(cudf::table_view const& source_table1,
                              cudf::table_view const& source_table2,
                              cudf::column_view const& gather_map1,
                              cudf::column_view const& gather_map2,
                              cudf::out_of_bounds_policy bounds_policy,
                              cudf::detail::negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  //CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

//   // create index type normalizing iterator for the gather_map
//   auto map_begin = indexalator_factory::make_input_iterator(gather_map);
//   auto map_end   = map_begin + gather_map.size();
//
//   if (neg_indices == negative_index_policy::ALLOWED) {
//     cudf::size_type n_rows = source_table.num_rows();
//     auto idx_converter     = cuda::proclaim_return_type<size_type>(
//       [n_rows] __device__(size_type in) { return in < 0 ? in + n_rows : in; });
//     return gather(source_table,
//                   thrust::make_transform_iterator(map_begin, idx_converter),
//                   thrust::make_transform_iterator(map_end, idx_converter),
//                   bounds_policy,
//                   stream,
//                   mr);
//   }
//   return gather(source_table, map_begin, map_end, bounds_policy, stream, mr);
    std::cout << "i am in gather from bucket chain hash join.cu" << std::endl;
    int n_match = gather_map1.size();
    int num_r = source_table1.num_rows();
    int num_s = source_table2.num_rows();
    int circular_buffer_size = std::max(num_r, num_s);
    SortHashGather shg(source_table1, source_table2, gather_map1, gather_map2, n_match, circular_buffer_size, 0, 15);
    auto result = shg.materialize_by_gather();;
    //auto result = SortHashJoin::materialize_by_gather(cudf::table_view source_table, cudf::column_view gather_map);
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

std::unique_ptr<table> gather(table_view const& source_table1,
                              table_view const& source_table2,
                              column_view const& gather_map1,
                              column_view const& gather_map2,
                              out_of_bounds_policy bounds_policy,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  //CUDF_FUNC_RANGE();

  auto index_policy = is_unsigned(gather_map1.type()) ? cudf::detail::negative_index_policy::NOT_ALLOWED
                                                     : cudf::detail::negative_index_policy::ALLOWED;

  return detail::gather(source_table1, source_table2, gather_map1, gather_map2, bounds_policy, index_policy, stream, mr);
}

// void print_table_view(cudf::table_view const& table) {
//     // Convert the table_view to a table
//     std::unique_ptr<cudf::table> table_copy = cudf::copy_table(table);
//
//     // Use CSV writer options to convert the table to a string
//     cudf::io::csv_writer_options write_options =
//         cudf::io::csv_writer_options::builder(cudf::io::sink_info()).include_header(true);
//
//     // Write the table to a CSV string
//     auto result = cudf::io::write_csv(write_options.set_table(table_copy->view()));
//
//     // Print the CSV string
//     std::cout << result << std::endl;
// }

}
