/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   libft.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/01 10:34:03 by cbouhadr          #+#    #+#             */
/*   Updated: 2025/05/10 13:38:42 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LIBFT_H
# define LIBFT_H

# include <unistd.h>
# include <stdlib.h>
# include <stdio.h>

//Partie 1
int		ft_isalpha(int c);
int		ft_isdigit(int c);
int		ft_isalnum(int c);
int		ft_isascii(int c);
int		ft_isprint(int c);
size_t	ft_strlen(const char *s);
void	*ft_memset(void *b, int c, size_t len);
void	ft_bzero(void *s, size_t n);
void	*ft_memcpy(void *dst, const void *src, size_t n);
void	*ft_memmove(void *dst, const void *src, size_t len);
size_t	ft_strlcpy(char *dst, const char *src, size_t dstsize); // a revoir
size_t	ft_strlcat(char *dst, const char *src, size_t dstsize); // a revoir
int		ft_toupper(int c);
int		ft_tolower(int c);
char	*ft_strchr(const char *s, int c);
char	*ft_strrchr(const char *s, int c);
int		ft_strncmp(const char *s1, const char *s2, size_t n);
void	*ft_memchr(const void *s, int c, size_t n);
int		ft_memcmp(const void *s1, const void *s2, size_t n);
char	*ft_strnstr(const char *haystack, const char *needle, size_t len);
void	*ft_calloc(size_t count, size_t size);
char	*ft_strdup(const char *s1);
int		ft_atoi(const char *str);

//Partie 2
char	*ft_substr(char const *s, unsigned int start, size_t len);
void	ft_striteri(char *s, void (*f)(unsigned int, char*));
char	*ft_strjoin(char const *s1, char const *s2, char sep);
char	*ft_strtrim(char const *s1, char const *set);
char	*ft_itoa(int n);
char	*ft_strmapi(char const *s, char (*f)(unsigned int, char));
void	ft_putchar_fd(char c, int fd);
void	ft_putstr_fd(char *s, int fd);
void	ft_putendl_fd(char *s, int fd);
void	ft_putnbr_fd(int n, int fd);
char	**ft_split(char const *s, char c);
size_t 	ft_get_split_len(char **split);
void	*ft_split_clean(char ***split);
void	ft_split_print(char **split);

typedef struct s_list
{
	void			*content;
	struct s_list	*next;
	
}	t_list;

t_list	*ft_lstnew(void *content);
void	ft_lstadd_front(t_list **lst, t_list *new);
int		ft_lstsize(t_list *lst);
t_list	*ft_lstlast(t_list *lst);
void	ft_lstadd_back(t_list **lst, t_list *new);
void	ft_lstdelone(t_list *lst, void (*del)(void*));
void	ft_lstclear(t_list **lst, void (*del)(void*));
void	ft_lstiter(t_list *lst, void (*f)(void *));
t_list	*ft_lstmap(t_list *lst, void *(*f)(void *), void (*del)(void *));

// personal add
int		ft_isset(char c, const char *set);
void	ft_swap_ptr(void *p1, void *p2, int len);
int		ft_cmp_char(void *p1, void *p2);
int		ft_cmp_int(void *p1, void *p2);
void	ft_swap_ptr(void *p1, void *p2, int size);
void	ft_bubble_sort(void *arr, int nb, int t,
			int (*cmp)(void *p1, void *p2));
int		ft_is_sort(void *arr, int nb, int t, int (*cmp)(void *p1, void *p2));
void	ft_generate_number(int *arr, int size);
void	ft_print_bit_8(int n);
void	ft_print_bit_16(int n);
void	ft_print_bit_32(int n);
void	ft_print_bit_64(int n);
int		ft_index_of_int(int *arr, int len, int n);
int		ft_index_of_c(const char *arr, char c);
void 	ft_generate_number(int *arr, int size);
int		ft_get_max_value(int *arr, int size);
void    ft_split_quick_sort(char **split, int end, int (*cmp)(const char *,const char *, size_t len));
int 	ft_is_split_sort(char **split);
void	ft_swap_str(char **str_1, char **str_2);
int 	ft_is_split_sort(char **split);
int 	ft_strlen_longest(char *str_1, char *str_2);
char 	*ft_skip_space_and_quote(char *str);
int		ft_isquote(char c);
int 	ft_nbr_of_word(const char *str);

#endif