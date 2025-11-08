/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_bublesort.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/27 12:00:52 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/29 17:00:51 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

void	ft_bubble_sort(void *arr, int nb, int t, int (*cmp)(void *p1, void *p2))
{
	int				i;
	int				j;
	unsigned char	*tab;

	i = 0;
	tab = (unsigned char *)arr;
	while (i < (nb - 1) * t)
	{
		j = i + t;
		if (cmp(&tab[i], &tab[j]) > 0)
		{
			ft_swap_ptr((void *)&tab[i], (void *)&tab[j], t);
			i = 0;
		}
		else
			i += t;
	}
}

// int main()
// {
// 	int arr[500];

// 	ft_generate_number(arr, 500);

// 	int len = 500;
// 	ft_bubble_sort(arr,len, sizeof(int), ft_cmp_int);
// 	if(!ft_is_sort(arr,500,4,ft_cmp_int))
// 		printf("le tableau est trie\n");
// 	int arr2[] = {-1, -3};
// 	printf("%d\n", ft_cmp_int(&arr2[0], &arr2[1]));
// 	return (0);
// }
