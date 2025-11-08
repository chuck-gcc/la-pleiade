/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strlcat.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/01 15:45:51 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/10/12 17:57:43 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

size_t	ft_strlcat(char *dst, const char *src, size_t dstsize)
{
	size_t	dst_len;
	size_t	src_len;
	size_t	i;
	size_t	reste;

	if (!dstsize && src)
		return (ft_strlen(src));
	dst_len = ft_strlen(dst);
	src_len = ft_strlen(src);
	if (dstsize <= dst_len)
		return (dstsize + src_len);
	reste = dstsize - dst_len;
	i = 0;
	while (i < reste - 1 && src[i])
	{
		dst[dst_len + i] = src[i];
		i++;
	}
	if (i < reste)
		dst[dst_len + i] = '\0';
	return (dst_len + src_len);
}
