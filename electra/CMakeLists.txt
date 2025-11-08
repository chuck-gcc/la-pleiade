cmake_minimum_required(VERSION 3.23)

project("electra")

file(GLOB SRCS srcs/*.c srcs/vm_managment/*.c srcs/offer_managment/*.c)
add_executable(electra ${SRCS})



#librairie libvirt
set(LIBVIRT_LIBRARY /usr/lib/x86_64-linux-gnu/libvirt.so)
set(LIBXML /usr/local/lib/libxml2.so)
set(LIBFT /home/cc/gpu_lab/pleiades/merope/42/libft/libft.a)

# header de la lib libvirt
set(VIRT /usr/include/libvirt)
set(LIBXML_HEADER usr/include/libxml)
set(LIBFT_HEADER /home/cc/gpu_lab/pleiades/merope/42/libft/libft.h)

#link the include
target_include_directories(electra PRIVATE ${VIRT})
target_include_directories(electra PRIVATE ${LIBXML_HEADER})
target_include_directories(electra PRIVATE ${LIBFT_HEADER})

#link the librarie
target_link_libraries(electra PRIVATE ${LIBVIRT_LIBRARY})
target_link_libraries(electra PRIVATE ${LIBXML})
target_link_libraries(electra PRIVATE ${LIBFT})



