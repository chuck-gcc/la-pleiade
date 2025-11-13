file(GLOB SRCS *.c vm_managment/*.c offer_managment/*.c)

add_executable(electra ${SRCS})

target_compile_options(electra PRIVATE -Wall -Wextra -Werror)

#librairie libvirt
set(LIBVIRT_LIBRARY /usr/lib/x86_64-linux-gnu/libvirt.so)
set(LIBXML /usr/lib/x86_64-linux-gnu/libxml2.so.2)
set(LIBFT /home/cc/gpu_lab/pleiades/merope/42/libft/libft.a)

# header de la lib libvirt
set(VIRT /usr/include/libvirt)
set(LIBXML_HEADER /usr/include/libxml)
set(LIBFT_HEADER /home/cc/gpu_lab/pleiades/merope/42/libft/libft.h)

#link the include
target_include_directories(electra PRIVATE ${VIRT}${LIBXML_HEADER}${LIBFT_HEADER})

#link the librarie
target_link_libraries(electra PRIVATE ${LIBVIRT_LIBRARY} ${LIBXML} ${LIBFT})



