/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strlen_longest.c                                :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/12 09:37:10 by cw3l              #+#    #+#             */
/*   Updated: 2025/04/25 11:41:29 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "libft.h"

int ft_strlen_longest(char *str_1, char *str_2)
{
    size_t len_1;
    size_t len_2;

    
    len_1 = ft_strlen(str_1);
    len_2 = ft_strlen(str_2);

    if(len_1 > len_2)
        return(len_1);
    return(len_2);
}
