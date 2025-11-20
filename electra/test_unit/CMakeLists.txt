    cmake_minimum_required(VERSION 3.28)

    project(testation)

    file(GLOB SRCS  ../srcs/vm_managment/*.c ../srcs/offer_managment/*.c)
    file(GLOB GETLINE_SRCS ../get_next_line/get_next_line_bonus.c ../get_next_line/get_next_line_utils_bonus.c)

    add_executable(test_date test_date.c ${SRCS})
    add_executable(test_date2 test_date2.c ${SRCS})
    add_executable(test_serveur test_serveur.c ${GETLINE_SRCS})

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
    target_include_directories(test_serveur PRIVATE  ${LIBFT_HEADER} )

    target_link_libraries(test_serveur PRIVATE ${LIBFT})
    target_link_libraries(test_date PRIVATE ${LIBVIRT_LIBRARY} ${LIBXML}  ${LIBFT})
    target_link_libraries(test_date2 PRIVATE ${READLINE_LIB}  ${LIBVIRT_LIBRARY} ${LIBXML}  ${LIBFT})

    add_test(NAME DateTest COMMAND test_date)
    add_test(NAME ServeurTest COMMAND test_serveur)
    add_test(NAME DateTest2 COMMAND test_date2)


