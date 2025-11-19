    cmake_minimum_required(VERSION 3.28)

    project(testation)

    file(GLOB SRCS  ../srcs/vm_managment/*.c ../srcs/offer_managment/*.c)

    add_executable(test_date test_date.c ${SRCS})
    add_executable(test_date2 test_date2.c ${SRCS})

    set(LIBFT /home/cc/gpu_lab/pleiades/merope/42/libft/libft.a)
    set(LIBFT_HEADER /home/cc/gpu_lab/pleiades/merope/42/libft)
    
    set(READLINE_HEADER /usr/local/include/readline)
    set(READLINE_LIB /usr/share/readline)

    set(LIBVIRT_LIBRARY /usr/lib/x86_64-linux-gnu/libvirt.so)
    set(VIRT /usr/include/libvirt)
    
    set(LIBXML /usr/lib/x86_64-linux-gnu/libxml2.so.2)
    
    

    set(LIBXML_HEADER /usr/include/libxml)

    target_include_directories(test_date PRIVATE ${VIRT} ${LIBXML_HEADER} ${LIBFT_HEADER} ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(test_date2 PRIVATE ${READLINE_HEADER}  ${VIRT} ${LIBXML_HEADER} ${LIBFT_HEADER} ${PROJECT_SOURCE_DIR}/include)

    target_link_libraries(test_date PRIVATE ${LIBVIRT_LIBRARY} ${LIBXML}  ${LIBFT})
    target_link_libraries(test_date2 PRIVATE ${READLINE_LIB}  ${LIBVIRT_LIBRARY} ${LIBXML}  ${LIBFT})

    add_test(NAME DateTest COMMAND test_date)
    add_test(NAME DateTest2 COMMAND test_date2)


