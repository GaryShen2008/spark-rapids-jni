/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_utilities.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

template <std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>> (*join_impl)(
            cudf::table_view const& left_keys,
            cudf::table_view const& right_keys,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr),
          cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK>
std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    join_impl(left_selected, right_selected, compare_nulls, stream, mr);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::inner_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}


void in_copy(int** arr, cudf::table_view table, int index){
   // Get the column_view for the first column (index 0) because we only support single key join now.
   cudf::column_view first_column = table.column(index);
   //std::cout << first_column.size() << std::endl;
   // Get the type of the first column.
   cudf::data_type dtype_r = first_column.type();
   const void* data_ptr_r;
   if (dtype_r.id() == cudf::type_id::INT32) {
      // The column type is INT32
      data_ptr_r = static_cast<const void*>(first_column.data<int32_t>());
      // Proceed with your INT32-specific logic here
      } else {
        // Handle other data types or throw an error if INT32 is required
        throw std::runtime_error("R key type not supported");
   }
   *arr = const_cast<key_t*>(reinterpret_cast<const key_t*>(data_ptr_r));
}


unsigned generate_random_unsigned() {
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<unsigned> distr; // Define the range

    return distr(gen); // Generate the random number
}


void printTable(cudf::table_view table){
    int N = table.num_rows();
    int M = table.num_columns();
    for(int i = 0; i < M; i++){
        int* col_dev {nullptr};
        in_copy(&col_dev, table, i);
        int* row_host = new int[N];
        cudaError_t cudaStatus = cudaMemcpy(row_host, col_dev, sizeof(int)*N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
          std::string errorMsg = "cudaMemcpy failed";
          errorMsg += cudaGetErrorString(cudaStatus);
          std::cerr << "CUDA Error: " << errorMsg << std::endl;
          throw std::runtime_error(errorMsg);
        }
        std::cout << "col" << i << ":";
        for (long j = 0; j < N; ++j) {
          std::cout << row_host[j] << " ";
        }
        delete[] row_host;
        std::cout << std::endl;
    }
}

void test_cudf(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  data_profile profile;
  profile.set_cardinality(100000); // Increase to reduce repetition
  profile.set_avg_run_length(1);
  profile.set_distribution_params<int>(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 1000);
  auto const table = create_random_table({cudf::type_id::INT32, cudf::type_id::INT32}, row_count{n_rows}, profile, generate_random_unsigned());
  std::cout << table->num_rows() << std::endl;
  std::cout << table->num_columns() << std::endl;
  const auto left = table->view();
  auto result = inner_join(left, left, {0}, {0});
  std::cout << result->num_rows();
  std::cout << std::endl;
  printTable(result->view());
}

NVBENCH_BENCH(test_cudf)
  .set_name("test cudf Only")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 2 * 1024 * 1024});

