using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

}

std::unique_ptr<table_view> join_gather_maps(table_view const& left_table,
                                        table_view const& right_table,
                                        bool compare_nulls_equal,
                                        rmm::device_async_resource_ref mr)
{
    CUDF_FUNC_RANGE();
    return detail::join_gather_maps(
        left_table,
        right_table,
        compare

    );
}
}
