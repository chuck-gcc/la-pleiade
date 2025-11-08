/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strjoin.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/10/02 10:43:46 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/10/16 14:16:10 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

char	*ft_strjoin(char const *s1, char const *s2, char separateur)
{

	char	*str;
	size_t	i;
	size_t	j;
	int sep;

	i = 0;
	j = 0;

	if (!s1 && !s2)
		return (NULL);
	if(separateur > 0)
		sep = 1;
	else
		sep = 0;
	str = malloc(sizeof(char) * (ft_strlen(s1) + ft_strlen(s2) + sep + 1));
	if (!str)
		return (NULL);	
	while (s1 && s1[j])
		str[i++] = s1[j++];
	if(sep)
		str[i++] = separateur;
	j = 0;
	while (s2 && s2[j])
		str[i++] = s2[j++];
	str[i] = '\0';
	return (str);
}

