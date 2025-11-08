/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_split_quick_sort.c                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/02 17:07:08 by cw3l              #+#    #+#             */
/*   Updated: 2025/04/04 12:03:03 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

static int ft_split_partition(char **split, int end, int (*cmp)(const char *,const char *, size_t len))
{
    int i;
    int j;
    char *pivot;
    
    i = 0;
    j = end - 1;
    pivot = split[end / 2];
    while (i < j)
    {
        while (cmp(split[i], pivot, ft_strlen(split[i])) < 0)
            i++;
        while (cmp(split[j], pivot, ft_strlen(split[j])) > 0)
            j--;
        if(cmp(split[i], split[j], ft_strlen(split[i])) > 0)
            ft_swap_str(&split[i], &split[j]);
    }
    return(i);
}

void   ft_split_quick_sort(char **split, int end, int (*cmp)(const char *,const char *, size_t len))
{
    int i;

    if(end <= 1)
        return ;
    i = ft_split_partition(split,end,cmp);
    ft_split_quick_sort(split,i,cmp);
    ft_split_quick_sort(&split[i + 1], end - 1 - i , cmp);

}