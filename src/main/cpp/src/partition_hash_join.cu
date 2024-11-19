#include "bucket_chain_hash_join.hpp"

namespace spark_rapids_jni {
namespace detail {
namespace {

} // namespace

void partition(){
    allocate_mem(&r_offsets, false, sizeof(int)*n_partitions, stream, mr);
    allocate_mem(&s_offsets, false, sizeof(int)*n_partitions, stream, mr);

    key_t* rkeys  {nullptr};
    key_t* skeys  {nullptr};

    get_table_column(rkeys, rkeys_partitions_tmp, rkeys_partitions, (key_t*)rvals_partitions_rkeys_partitions);
}

void join_copartitions(){

}

// std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
//           std::unique_ptr<rmm::device_uvector<size_type>>>
// join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr){
//
//
//
// }


template<typename KeyT, typename ValueT>
void partition_pairs(KeyT* keys, ValueT* values, KeyT* keys_out, ValueT* values_out, int* offsets, const int num_items){
    SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits, stream, mr);
    ssp.process();
}


} // namespace detail

partition_hash_join::~partition_hash_join() = default;

template<typename T>
void get_table_column(T** arr, cudf::table_view &table, int index){
    cudf::column_view first_column = table.column(index);
    cudf::data_type dtype_r = first_column.type();

    const void* data_ptr_r;
    if(dtype_r.id() == cudf::type_id::INT32){
        data_ptr_r = static_cast<const void*>(first_column.data<int32_t>());
    } else {
        throw std::runtime_error("data type column get not supported.");
    }

    *arr = const_cast<key_t*>(reinterpret_cast<const key_t*>(data_ptr_r));
}

partition_hash_join(cudf::table_view &r_in, cudf::table_view &s_in, int first_bit, int radix_bits, int circular_buffer_size, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
    // this is for getting the num of rows of table r and s.
    nr = static_cast<int>(r_in.num_row());
    ns = static_cast<int>(s_in.num_row());

    // why there is 2048 more rows? TODO: figure out the reason.
    allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048), stream, mr);
    allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048), stream, mr);

    allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048), stream, mr);
    allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048), stream, mr);

    allocate_mem(&rkeys_partitions_tmp, false, sizeof(key_t)*(nr+2048), stream, mr);
    allocate_mem(&skeys_partitions_tmp, false, sizeof(key_t)*(ns+2048), stream, mr);


    fill_sequence<<<num_tb(nr), 1024>>>((int*)(rkeys_partitions_tmp), 0, nr);
    fill_sequence<<<num_tb(ns), 1024>>>((int*)(skeys_partitions_tmp), 0, ns);

}

partition_hash_join(cudf::table_view &r_in, cudf::table_view &s_in, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr){

}


} // namespace spark_rapids_jni