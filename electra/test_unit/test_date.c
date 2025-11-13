#include "../include/electra/test_unit.h"

int main(void)
{
    t_resa resa;
    char **resa_split;
    char *args = "2027-07-13T07:12:12 2 3";

    resa_split = ft_split(args, 32);
    if(!resa_split)
        return(1);

    resa = *iso_date_to_resa(resa_split[0], resa_split[1], resa_split[2]);

    assert(resa.offer_id == 3);


    ft_split_print(resa_split);
    ft_split_clean(&resa_split);
    printf("test psser avec succes\n");
    return(0);
}