/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strnstr.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/01 14:54:18 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/10/12 16:49:51 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

char	*ft_strnstr(const char *haystack, const char *needle, size_t len)
{
	size_t	i;
	size_t	needle_len;

	i = 0;
	needle_len = ft_strlen(needle);
	if (needle[0] == '\0')
		return ((char *)haystack);
	if (needle && len == 0)
		return (NULL);
	while (haystack[i])
	{
		if (needle[0] == haystack[i] && i + needle_len <= len)
			if (!ft_strncmp(&haystack[i], needle, needle_len))
				return ((char *)&haystack[i]);
		i++;
	}
	return (NULL);
}
