#include "vm_managment/vm.h"
#include "offer_managment/offer.h"

int main(void)
{
    t_offre **list;

    list =  get_offer_list();
    dislay_offer(list);
    clean_list_offer(list);
    free(list);
    // virConnectPtr co;

    // co = host_connection();
    // if(!co)
    //     return(1);
    // list_actif_domain(co);
    // //create_vm(co);

    // get_ressources(co);
    // virConnectClose(co);
    return(0);
}