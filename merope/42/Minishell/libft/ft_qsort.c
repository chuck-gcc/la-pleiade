/******************************************************************************/
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_qsort.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ast <ast@student.42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/27 11:49:47 by cbouhadr          #+#    #+#             */
/*   Updated: 2025/05/03 03:17:08 by ast              ###   ########.fr       */
/*                                                                            */
/******************************************************************************/

#include "libft.h"

// void ft_print(int *arr, int len)
// {
// 	int j;
	
// 	j = 0;
// 	while (j < len)
// 	{
// 		printf(" %d", (int)arr[j]);
// 		j++;
// 	}
// 	printf("\n");
// }
// int	ft_partition(void *arr, int end, int t, int (*cmp)(void *, void *))
// {
// 	int				idx_pivot;
// 	unsigned char 	*tab;
// 	int				i;
// 	int 			j;
	
// 	i = 0;
// 	j = 0;
// 	tab = (unsigned char *)arr;

// 	idx_pivot = (end - 1) * t;
// 	while (j < (end - 1) * t)
// 	{
// 		if(cmp(&tab[i], &tab[j]) > 0)
// 			ft_swap_ptr(&tab[i], &tab[j], t);
// 		if(cmp(&tab[i], &tab[idx_pivot]) > 0)
// 			j += t;
// 		else
// 		{	
// 			i += t;
// 			j += t;
// 		}
// 	}
	
// 	return ((i / t));
// } 

// void	ft_qsort(void *base, int nel, int width, int (*cmp)(void *, void *))
// {
// 	int i;
// 	int start;
// 	int end;

// 	i = 0;
// 	start = i;
// 	end = (nel * width) - width;
	
// 	if (nel <= 1)
// 	 	return;
	
// 	i = ft_partition(base, nel, width, cmp);
// 	ft_qsort(base,i,width,cmp);
// 	ft_qsort(&base[i], nel - i, width, cmp);
	
// }

// int  main(void)
// {
// 	//int arr[] = {42, -17, 85, 3, -9, 61, 28, 70, -35, 14, 99, -22, 56, 0, 1};
// 	int arr[] = {14, 99, -22, 56, 77, 55};

// 	int size = sizeof(arr) / sizeof(arr[0]);
 	//ft_qsort(arr,size,4,ft_cmp_char);
	
// 	ft_print(arr,size);
// 	return (0);
// }
