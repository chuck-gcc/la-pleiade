/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_get_max_value.c                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/12/14 11:59:12 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/12/14 11:59:33 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

int	ft_get_max_value(int *arr, int size)
{
	int	i;
	int	high;

	i = 1;
	high = arr[0];
	while (i < size)
	{
		if (arr[i] > high)
			high = arr[i];
		i++;
	}
	return (high);
}
