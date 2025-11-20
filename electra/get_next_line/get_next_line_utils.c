/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   get_next_line_utils.c                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/25 06:52:55 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/10 12:45:30 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "get_next_line.h"

int	ft_len_index_of(char *str, char mode)
{
	int	i;

	i = 0;
	if (!str)
	{
		if (mode == 'i')
			return (-1);
		else
			return (0);
	}
	while (str && str[i])
	{
		if (str[i] == 10 && mode == 'i')
			return (i);
		i++;
	}
	if (mode == 'i')
		return (-1);
	return (i);
}

char	*ft_substr(char *str, int start, int len)
{
	char	*sub;
	int		str_len;
	int		i;

	if (!str)
		return (NULL);
	sub = malloc(sizeof(char) * (len + 1));
	if (!sub)
		return (NULL);
	str_len = ft_len_index_of(str, 'l');
	i = 0;
	if (start > str_len)
	{
		free(sub);
		return (NULL);
	}
	while (str[start] && i < len)
	{
		sub[i] = str[start];
		i++;
		start++;
	}
	sub[i] = '\0';
	return (sub);
}

char	*ft_strdup(char *str)
{
	int		len;
	char	*new_str;
	int		i;

	if (!str)
		return (NULL);
	len = ft_len_index_of(str, 'l');
	i = 0;
	if (len == 0)
		return (NULL);
	new_str = malloc(sizeof(char) * (len + 1));
	if (!new_str)
		return (NULL);
	while (str[i])
	{
		new_str[i] = str[i];
		i++;
	}
	new_str[i] = '\0';
	return (new_str);
}

char	*ft_strjoin(char *s1, char *s2)
{
	int		len;
	int		i;
	int		j;
	char	*new_str;
	char	*tmp;

	if (!s1 && !s2)
		return (NULL);
	len = ft_len_index_of(s1, 'l') + ft_len_index_of(s2, 'l');
	new_str = malloc(sizeof(char) * (len + 1));
	tmp = s1;
	if (!new_str)
		return (NULL);
	i = 0;
	j = 0;
	while (tmp && tmp[j])
		new_str[i++] = tmp[j++];
	j = 0;
	tmp = s2;
	while (tmp && tmp[j])
		new_str[i++] = tmp[j++];
	new_str[i] = '\0';
	return (new_str);
}

char	*ft_strchr(char *str, char c)
{
	if (!str)
		return (NULL);
	while (*str)
	{
		if (*str == c)
			return (str);
		str++;
	}
	return (NULL);
}
