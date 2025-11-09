#include "vm_managment/vm.h"
#include "offer_managment/offer.h"

int main(void)
{
    t_offre **list;

    list =  get_offer_list();

    //dislay_offer_list(list);

    test_time(list);


    clean_list_offer(list);
    free(list);
    return(0);
}