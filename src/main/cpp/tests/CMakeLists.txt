#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_compile_options(${CMAKE_TEST_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${SPARK_RAPIDS_JNI_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_RAPIDS_JNI_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_TEST_NAME}
                PRIVATE "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}>"
                        "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}/src>")
    set_target_properties(
        ${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/gtests>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
    )
    target_link_libraries(${CMAKE_TEST_NAME} GTest::gtest_main GTest::gmock_main cudf::cudf
                                             cudf::cudftestutil spark_rapids_jni)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    install(
        TARGETS ${CMAKE_TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests
        EXCLUDE_FROM_ALL
    )
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

ConfigureTest(ROW_CONVERSION
    row_conversion.cpp)
