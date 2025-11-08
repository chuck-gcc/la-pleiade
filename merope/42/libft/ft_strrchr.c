/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strrchr.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/01 14:19:06 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/10/10 13:45:30 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

char	*ft_strrchr(const char *s, int c)
{
	size_t	i;
	int		last;

	i = 0;
	last = -1;
	while (s[i])
	{
		if ((char)s[i] == (char)c)
			last = i;
		i++;
	}
	if ((char)c == '\0')
		return ((char *)&s[i]);
	if (last >= 0)
		return ((char *)&s[last]);
	return (NULL);
}
