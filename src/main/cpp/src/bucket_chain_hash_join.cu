using namespace cudf;

namespace spark_rapids_jni {

namespace detail {


std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
              std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
               table_view const& right_input,
               null_equality compare_nulls,
               rmm::cuda_stream_view stream
               rmm::device_async_resource_ref mr){
    // Dummy implementation
    return {};
}

} // detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
              std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
               table_view const& right_input,
               null_equality compare_nulls,
               rmm::cuda_stream_view stream
               rmm::device_async_resource_ref mr){
    return detail::inner_join(left_input, right_input, stream, mr);
}
}
