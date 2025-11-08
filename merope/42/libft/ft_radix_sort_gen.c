/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_radix_sort_gen.c                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/12/17 08:35:30 by cw3l              #+#    #+#             */
/*   Updated: 2024/12/17 09:08:13 by cw3l             ###   ########.fr       */
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

void ft_raidx_sort_gen(int *arr, int len, int base)
{
    int bit;
    int digit;
    int i;
    int cumulatif[base];

    i = 0;
    bit = 0;
    while (i < len)
    {
        digit =  ((arr[i] >> bit) & 1);
        cumulatif[digit]++;
        i++;
    }
    ft_print_arr(cumulatif, 2);

    i = 1;
    while (i < 2)
    {
        cumulatif[i] += cumulatif[i - 1];
        i++;
    }
    //ft_print_arr(cumulatif, 2);
    
}

int main(void)
{
    int arr[] = {10, 11, -1, -5, 7};

    int len = sizeof(arr) / sizeof(arr[0]);

    int i;

    i = 0;
    while (i < len)
    {
        ft_print_bit_8(arr[i]);
        i++;
    }

    printf("\n");
    ft_raidx_sort_gen(arr, len, 2);
    return(0);
}