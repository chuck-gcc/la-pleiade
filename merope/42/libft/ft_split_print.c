/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_split_print.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/28 14:16:19 by cbouhadr          #+#    #+#             */
/*   Updated: 2025/04/25 15:45:49 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"
#include <assert.h>

void	ft_split_print(char **split)
{
	
	int i;

	i = 0;
	while (split[i])
	{
		write(1, split[i], ft_strlen(split[i]));
		write(1, "\n", 1);
		i++;
	}
}
