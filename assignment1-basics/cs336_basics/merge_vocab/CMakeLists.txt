cmake_minimum_required(VERSION 3.10)

project(merge_vocab)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# submodules
# add_subdirectory(third_party/fmt)

# headers
include_directories(
    # /home/ztzhu/.conda/envs/exp/bin/python
    /home/ztzhu/AI/CS336/assignment1-basics/.venv/lib/python3.12/site-packages/pybind11/include
    /home/ztzhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12
    # ${CMAKE_SOURCE_DIR}/third_party/fmt/include
)

# libs
link_directories(/home/ztzhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib ${CMAKE_SOURCE_DIR}/lib)

# files
set(
    SRC_LIST
    "src/main.cpp"
)

# target
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_link_libraries(
    ${PROJECT_NAME} PRIVATE
    # /home/ztzhu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
)
set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}.cpython-312-x86_64-linux-gnu
)

set(FILENAME ${PROJECT_NAME}.cpython-312-x86_64-linux-gnu.so)

add_custom_target(
    build
    COMMAND mv lib/lib${FILENAME} lib/../../${FILENAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
