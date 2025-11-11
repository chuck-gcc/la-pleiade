#include "vm_managment/vm.h"
#include "offer_managment/offer.h"

struct tm date_obj(t_date date)
{

    struct tm t;

    t.tm_mday = (int)date.day;
    t.tm_mon = (int)date.mount;
    t.tm_year = (int)date.year;
    t.tm_hour = (int)date.hour;
    t.tm_min = (int)date.min;
    t.tm_sec = (int)date.sec;
    return(t);
}

int main(void)
{
    t_offre **list;
    t_resa **resa_tree;
    list =  get_offer_list();

    
    time_t now;
    now = time(NULL);
    struct tm *t = localtime(&now);
    

    resa_tree = malloc(sizeof(t_resa *));
    if(!resa_tree){perror("malloc"); return(1);}

    *resa_tree = NULL;
    create_resa(t, 2, 0,resa_tree);
    
    display_resa(*resa_tree);
    free(list);
    free(resa_tree);
    return(0);
}