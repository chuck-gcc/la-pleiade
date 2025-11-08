cmake_minimum_required(VERSION 3.5)

project(cpp_42)

set(SRCS cpp-00/ex00.cpp)
file(GLOB SRCS_2 "cpp-01/*.cpp")

add_executable(megaphone ${SRCS})
add_executable(phone ${SRCS_2})

