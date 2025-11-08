/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_radix_sort.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/12/16 19:32:15 by cw3l              #+#    #+#             */
/*   Updated: 2025/04/12 10:51:58 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"


void ft_print_arr(int *arr, int len)
{
	int i;

	i = 0;
	while (i < len)
		printf("%d ", arr[i++]);
}

void	ft_arr_cpy(int *src, int *dst, int len)
{
	int i;

	i = 0;
	while (i < len)
	{
		dst[i] = src[i];
		i++;
	}
	
}

int ft_get_max(int *arr, int len)
{
	int i;
	int max;

	i = 0;
	if (!arr || len == 0)
		return (-1);
	while(i < len)
	{
		if(arr[i] > max)
			max = arr[i];
		i++;
	}
	return(max);
}

void	ft_count_digit(int *arr, int *cumulatif, int len, int exp)
{
	int i;
	int digit;

	i = 0;
	while (i < len)
	{
		digit = (arr[i] / exp) % 10;
		cumulatif[digit]++;
		i++;
	}
}
void ft_cumulatif_arr(int *cumulatif_arr)
{
	int i;
	
	i = 1;
	while (i < 10)
	{
		cumulatif_arr[i] += cumulatif_arr[i - 1];
		i++;
	}
}
void ft_sort(int *arr, int *tmp ,int len, int exp)
{
	int digit;
	int h;
	int cumulatif_arr[10];

	// etape zero, mettre a initialiser a zero le tableau cumulatif.
	ft_bzero(cumulatif_arr, 10 * 4);
	//etape 1 etablire le compte pour chaque digit.
	ft_count_digit(arr, cumulatif_arr, len, exp);
	ft_cumulatif_arr(cumulatif_arr);
	h = len - 1;
	while (h >= 0)
	{
		digit = (arr[h] / exp) % 10;
		tmp[cumulatif_arr[digit] - 1] = arr[h];
		cumulatif_arr[digit]--;
		h--;
	}
	ft_memcpy(arr,tmp,len * sizeof(int));
}

void ft_radix_sort(int *arr, int len)
{
	int exp;
	int max;
	int *tmp;

	tmp = malloc(len * sizeof(int));
	if(!tmp)
		return ;
	exp = 1;
	max = ft_get_max(arr, len);
	if (max == -1)
		return ; 
	while (max / exp > 0)
		ft_sort(arr, tmp ,len, exp *= 10);
	free(tmp);
}

int main()
{
	int arr[123] = {
        100, 1000, 1, 33, 54, 77, 3455, 22, 55, 63, 112, 210, 500, 781, 900, 1234, 5678, 4321, 8765,
        10000, 2, 5000, 111, 201, 341, 442, 543, 632, 732, 832, 934, 103, 2100, 5001, 6400, 7500, 8230, 
        9300, 10111, 1199, 2340, 4560, 6789, 8321, 9456, 10500, 1500, 2500, 3500, 4500, 5500, 6512, 7523, 
        8534, 9545, 10656, 11767, 12878, 13989, 15012, 16123, 17234, 18345, 19456, 20567, 21678, 22789, 
        23890, 24901, 26012, 27123, 28234, 29345, 30456, 31567, 32678, 33789, 34890, 35901, 37012, 38123, 
        39234, 40345, 41456, 42567, 43678, 44789, 45890, 46901, 48012, 49123, 50234, 51345, 52456, 53567, 
        54678, 55789, 56890, 57901, 59012, 60123, 61234, 62345, 63456, 64567, 65678, 66789, 67890, 68901, 
        70012, 71123, 72234, 73345, 74456, 75567, 76678, 77789, 78890, 79901, 81012, 82123, 83234, 84345
    };
	int len = sizeof(arr) / sizeof(arr[0]);

	
	ft_radix_sort(arr, len);
	//ft_print_arr(arr, len);
	if(ft_is_sort(arr,123,4,ft_cmp_int))
		printf("vle tableau est bien trié\n");

	int p =  10;
	int q = -10;

	ft_print_bit_8(p);
	ft_print_bit_8(q);
	
	return(0);
}