/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_swap_ptr.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/27 14:17:44 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/27 14:52:20 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

void	ft_swap_ptr(void *p1, void *p2, int size)
{
	void	*tmp;

	if (p1 && p2)
	{
		tmp = malloc(size + 1);
		if (!tmp)
			return ;
		ft_memcpy(tmp, p1, size);
		ft_memcpy(p1, p2, size);
		ft_memcpy(p2, tmp, size);
	}
}
