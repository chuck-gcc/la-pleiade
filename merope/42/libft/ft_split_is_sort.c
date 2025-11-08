/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_split_is_sort.c                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/02 17:07:41 by cw3l              #+#    #+#             */
/*   Updated: 2025/04/02 17:09:02 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

int ft_is_split_sort(char **split)
{
    int i;
    int j;
    int split_len;

    i = 0;
    split_len = ft_get_split_len(split);
    while (i < split_len - 1)
    {
        j = i + 1;
        if(ft_strncmp(split[i], split[j], ft_strlen(split[i])) > 0)
            return(0);
        i++;
    }
    return(1);
    
}
