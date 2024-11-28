#include "bucket_chain_hash_join.hpp"

namespace spark_rapids_jni {
namespace detail {
namespace {

} // namespace

void partition(){
    // n partitions
    allocate_mem(&r_offsets, false, sizeof(int)*n_partitions, stream, mr);
    allocate_mem(&s_offsets, false, sizeof(int)*n_partitions, stream, mr);

    key_t* rkeys  {nullptr};
    key_t* skeys  {nullptr};

    get_table_column(&rkeys, r, 0);
    get_table_column(&skeys, s, 0);

    // consider using iterator here
    partition_pairs(rkeys, rkeys_partitions_tmp, rkeys_partitions, (key_t*)rvals_partitions, r_offsets,n nr);

    partition_pairs(skeys, skeys_partitions_tmp, skeys_partitions, (key_t*)svals_partitions, s_offsets, ns);

    release_mem(rkeys_partitions_tmp, sizeof(key_t)*(nr+2048), stream, mr);
    release_mem(skeys_partitions_tmp, sizeof(key_t)*(ns+2048), stream, mr);
    // a chunk contains n / n_partitions
    generate_work_units<<<num_tb(n_partitions,512),512>>>(r_offsets, s_offsets, r_work, s_work, total_work, n_partitions, threshold);

}

void join_copartitions(){
    CHECK_LAST_CUDA_ERROR();
    constexpr int NT = 512;
    constexpr int VT = 4;

    size_t sm_bytes = (bucket_size + 512) * (sizeof(key_t) + sizeof(int) + sizeof(int16_t)) + // elem, payload and next resp.
                   (1 << LOCAL_BUCKETS_BITS) * sizeof(int32_t) + // hash table head
                    + SHUFFLE_SIZE * (NT/32) * (sizeof(key_t) + sizeof(int)*2);

    auto join_fn = join_copartitions_arr_v1<NT, VT, LOCAL_BUCKETS_BITS, SHUFFLE_SIZE, key_t, int, key_t>;
    cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
    join_fn<<<n_partitions, NT, sm_bytes>>>(
                                           rkeys_partitions, rvals_partitions,
                                           skeys_partitions, svals_partitions,
                                           r_work, s_work,
                                           total_work,
                                           radix_bits, 4096,
                                           d_n_matches,
                                           r_match_idx, s_match_idx,
                                            circular_buffer_size);

    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(&n_matches, d_n_matches, sizeof(n_matches), cudaMemcpyDeviceToHost);
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
: r(r_in)
, s(s_in)
, first_bit(first_bit)
, circular_buffer_size(circular_buffer_size)
, radix_bits(radix_bits)
{
    // this is for getting the num of rows of table r and s.
    nr = static_cast<int>(r_in.num_row());
    ns = static_cast<int>(s_in.num_row());

    // why there is 2048 more rows? TODO: figure out the reason.
    // allocated nr + 2048 elements
    allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048), stream, mr);
    allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048), stream, mr);

    allocate_mem(&rvals_partitions, false, sizeof(int32_t)*(nr+2048), stream, mr);
    allocate_mem(&svals_partitions, false, sizeof(int32_t)*(ns+2048), stream, mr);

    allocate_mem(&rkeys_partitions_tmp, false, sizeof(key_t)*(nr+2048), stream, mr);
    allocate_mem(&skeys_partitions_tmp, false, sizeof(key_t)*(ns+2048), stream, mr);

    // iterator for 0 to n - 1
    fill_sequence<<<num_tb(nr), 1024>>>((int*)(rkeys_partitions_tmp), 0, nr);
    fill_sequence<<<num_tb(ns), 1024>>>((int*)(skeys_partitions_tmp), 0, ns);

}

partition_hash_join(cudf::table_view &r_in, cudf::table_view &s_in, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr){

}


} // namespace spark_rapids_jni