cmake_minimum_required(VERSION 3.23)

project(electra)

enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory("srcs")

add_subdirectory("test_unit")





