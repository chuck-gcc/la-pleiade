CC = gcc
AR = ar -crs
HDRS = libft.h
NAME = libft.a
NAME_TEST = test
CFLAGS = -Wall -Werror -Wextra

SRCS=	ft_putchar_fd.c \
		ft_putstr_fd.c \
		ft_putendl_fd.c \
		ft_putnbr_fd.c	\
		ft_isalpha.c \
		ft_isdigit.c \
		ft_isalnum.c \
		ft_isascii.c \
		ft_isprint.c \
		ft_strlen.c \
		ft_memset.c \
		ft_bzero.c \
		ft_memcpy.c \
		ft_memmove.c \
		ft_strlcpy.c \
		ft_strlcat.c \
		ft_toupper.c \
		ft_tolower.c \
		ft_strchr.c \
		ft_strrchr.c \
		ft_memchr.c \
		ft_memcmp.c \
		ft_strnstr.c \
		ft_calloc.c \
		ft_strdup.c \
		ft_strmapi.c \
		ft_striteri.c \
		ft_strjoin.c \
		ft_strtrim.c \
		ft_split.c \
		ft_split_len.c \
		ft_split_clean.c \
		ft_split_print.c \
		ft_strncmp.c \
		ft_itoa.c \
		ft_atoi.c \
		ft_substr.c \
		ft_cmp_char.c \
		ft_cmp_int.c \
		ft_swap_ptr.c \
		ft_is_sort.c \
		ft_isset.c \
		ft_print_bit_8.c \
		ft_print_bit_16.c \
		ft_print_bit_32.c \
		ft_print_bit_64.c \
		ft_index_of_int.c \
		ft_index_of_char.c \
		ft_get_max_value.c \
		ft_split_quick_sort.c \
		ft_split_is_sort.c \
		ft_swap_str.c \
		ft_strlen_longest.c \
		ft_skip_space.c \
		ft_isquote.c \
		ft_nbr_of_word.c \
		ft_day_in_mount.c \

BONUS=	ft_lstnew_bonus.c \
		ft_lstadd_front_bonus.c \
		ft_lstsize_bonus.c \
		ft_lstlast_bonus.c \
		ft_lstadd_front_bonus.c \
		ft_lstadd_back_bonus.c \
		ft_lstdelone_bonus.c \
		ft_lstclear_bonus.c \
		ft_lstiter_bonus.c \
		ft_lstmap_bonus.c \
		
OBJS=${SRCS:.c=.o}
BONUS_OBJS=${BONUS:.c=.o}


all: ${NAME}

${NAME}: ${OBJS} ${HDRS}
	${AR} ${NAME} ${OBJS}


bonus: ${OBJS} ${BONUS_OBJS} ${HDRS}
	${AR} ${NAME} ${OBJS} ${BONUS_OBJS}

clean:
	rm -rf *.o *.dSYM

fclean: clean
	rm -f ${NAME} 
	rm -f ${NAME} 

re: fclean ${NAME}

# test
r:
	rm -f ${NAME_TEST}

test: r
	${CC} ${CFLAGS} *.c -o ${NAME_TEST}

t:  test
	./${NAME_TEST}

gdev:
	cd .. && git add .
	git commit -m $(NAME)/$(DATE) 
	git push

.phony: clean fclean re all bonus gdev