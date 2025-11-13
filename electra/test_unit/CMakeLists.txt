
add_executable(test_date test_date.c)
add_executable(test_date2 test_date2.c)

target_include_directories(test_date PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_date2 PRIVATE ${PROJECT_SOURCE_DIR}/include)


add_test(NAME DateTest COMMAND test_date)
add_test(NAME DateTest2 COMMAND test_date2)


