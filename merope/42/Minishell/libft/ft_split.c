/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_split.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/02 14:53:35 by cbouhadr          #+#    #+#             */
/*   Updated: 2025/03/28 23:55:34 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"
#include <assert.h>

static int	ft_clean_memory_exit(char ***split, int idx)
{
	int	i;

	i = 0;
	while (i < idx)
	{
		free(*split[i]);
		*split[i] = NULL;
		i++;
	}
	free(*split);
	*split = NULL;
	return (0);
}

int	ft_count_word(char const *str, char c)
{
	int	i;
	int	on;
	int	count;

	i = 0;
	on = 0;
	count = 0;
	if (!str[i])
		return (-1);
	while (str[i])
	{
		if (str[i] != c && on == 0)
		{
			count++;
			on = 1;
		}
		else if (str[i] == c)
			on = 0;
		i++;
	}
	return (count);
}

int	ft_get_word_len(char const *str, char c)
{
	int	i;

	i = 0;
	if (!str)
		return (0);
	while (str[i] && str[i] != c)
		i++;
	return (i);
}

int	ft_process_data(char **split, char const *str, char c)
{
	int	i;
	int	j;
	int	word_len;

	i = 0;
	j = 0;
	while (str[i])
	{
		if (str[i] != c)
		{
			word_len = ft_get_word_len(&str[i], c);
			split[j] = malloc(sizeof(char) * (word_len + 1));
			if (!split[j])
				return (ft_clean_memory_exit(&split, i));
			ft_strlcpy(split[j], &str[i], word_len + 1);
			i += word_len;
			j++;
		}
		else
			i++;
	}
	split[j] = NULL;
	return (1);
}

char	**ft_split(char const *str, char c)
{
	size_t	size_arr_word;
	char	**arr_str;

	size_arr_word = 0;
	if (!str)
		return (NULL);
	size_arr_word = ft_count_word(str, c);
	if ((int)size_arr_word == -1)
		return (NULL);
	arr_str = malloc((sizeof(char *) * (size_arr_word + 1)));
	if (!arr_str)
		return (NULL);
	if (!ft_process_data(arr_str, str, c))
	{
		free(arr_str);
		arr_str = NULL;
	}
	return (arr_str);
}
