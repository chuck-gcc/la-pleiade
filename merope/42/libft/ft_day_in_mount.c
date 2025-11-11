#include "libft.h"

#define JAN 1
#define FEV 2
#define MARS 3
#define APRIL 4
#define MAI 5
#define JUIN 6
#define JUIL 7
#define AOUT 8
#define SEPT 9
#define OCTO 10
#define NOV 11
#define DEC 12

unsigned int day_in_mount(t_date date)
{
    unsigned int mount;
    unsigned int year;

    mount = date.mount;
    year = date.year;
    if(mount == 0 || mount > 12)
        return(-1);
    if(mount == JAN
    || mount == MARS
    || mount == MAI
    || mount == JUIL
    || mount == AOUT
    || mount == OCTO
    || mount == DEC
    )
        return(31);
    else if (mount == FEV)
    {
        if(year % 4 == 0)
            return(29);
        else
            return(28);
    }
    else
        return(30);
}