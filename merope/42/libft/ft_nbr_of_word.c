/************************************/
/*                                  */
/*                                  */
/*                                  */
/*                                  */
/************************************/

#include "libft.h"

int ft_nbr_of_word(const char *str)
{
    char **split;
    int count;

    if(!str)
        return(-1);
    split = ft_split(str, 32);
    if(!split)
        return(-1);
    count = ft_get_split_len(split);
    ft_split_clean(&split);
    return(count);
}