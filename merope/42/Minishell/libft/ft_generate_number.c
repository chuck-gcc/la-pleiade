/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_random_int.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/29 11:51:51 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/12/09 16:34:55 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

static int ft_is_used(int *arr, int len, int value)
{
    int i;

    i = 0;
    while (i < len)
    {
        if(arr[i] == value)
            return (1);
        i++;
    }
    return (0);
}


void ft_generate_number(int *arr, int size)
{
    int i;
    int random_num;

    i = 0;
    while (i < size)
    {
       random_num =  random() % size + 1;
       if(!ft_is_used(arr, i + 1 ,random_num))
       {
            arr[i] = random_num;
            i++;
       }
    }
    
}