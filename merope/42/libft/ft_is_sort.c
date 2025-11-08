/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_is_sort.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/27 14:26:40 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/27 14:41:11 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_is_sort.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/27 14:26:40 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/27 14:41:11 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

int	ft_is_sort(void *arr, int nb, int t, int (*cmp)(void *p1, void *p2))
{
	int	i;
	int	j;
	unsigned char *tmp;
	
	if(!arr || nb < 2)
		return(-1);
	i = 0;
	tmp = (unsigned char *)arr;
	while (i < (nb * t) - t)
	{
		j = i + t;
		if (cmp(&tmp[i], &tmp[j]) > 0)
			return (1);
		i += t;
	}
	return (0);
}

