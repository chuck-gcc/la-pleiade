#include "../include/electra/test_unit.h"

typedef struct s_date_1
{
    int day;
    int mount;
    int year;

} t_date_1;

int get_date(t_date_1 date, char *iso_date, char sep)
{

    if(!*iso_date)
    {
        return(1);
    }
    if(*iso_date != sep)
        printf("%c", *iso_date);
    get_date(date, &*(iso_date + 1), sep);
}

int parse_iso_date(char *iso_date)
{
    char **date_split;
    char *date_part, *hour_part;
    t_date_1 date;

    if(!iso_date)
        return(1);

    date_split = ft_split(iso_date, 'T');
    if(!date_split)
        return(1);
    date_part = date_split[0];
    hour_part = date_split[1];
    ft_bzero(&date,sizeof(t_date_1));
    get_date(date,date_part,'-');
    return(0);    
}

int main(void)
{
    char **resa_split;
    char *args = "2027-12-31T07:12:12 2 3";

    resa_split = ft_split(args, 32);
    if(!resa_split)
        return(1);
    parse_iso_date(resa_split[0]);
    
    ft_split_clean(&resa_split);
    printf("test psser avec succes\n");
    return(0);
}