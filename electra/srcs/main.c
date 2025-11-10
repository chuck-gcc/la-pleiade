#include "vm_managment/vm.h"
#include "offer_managment/offer.h"

int main(void)
{
    t_offre **list;

    list =  get_offer_list();

    //dislay_offer_list(list);
    
    t_date date;

    date.mount = 2;
    date.year = 2028;
    unsigned int day = day_in_mount(date);
    
    printf("voici le nombre de jour %d pour mars 2025\n",day);
    //test_time(list);



    clean_list_offer(list);
    free(list);
    return(0);
}