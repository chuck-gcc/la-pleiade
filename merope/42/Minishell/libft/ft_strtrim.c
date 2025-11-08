/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strtrim.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/02 10:58:24 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/12/09 16:52:16 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

char	*ft_strtrim(char const *s1, char const *set)
{
	size_t	i;
	size_t	s_len;
	size_t	len;
	char	*str;

	i = 0;
	if (!s1 || !set)
		return (NULL);
	s_len = ft_strlen(s1);
	while (ft_isset(s1[i], set))
		i++;
	if (i == s_len)
		return (ft_strdup(""));
	while (ft_isset(s1[s_len - 1], set))
		s_len--;
	len = (s_len - i) + 1;
	str = malloc(sizeof(char) * len);
	if (!str)
		return (NULL);
	ft_strlcpy(str, &s1[i], len);
	return (str);
}
