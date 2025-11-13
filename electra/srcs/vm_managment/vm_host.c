#include "../include/electra/vm.h"

virConnectPtr host_connection(void)
{
    virConnectPtr co;

    co = virConnectOpen(NULL);
    if(!co)
    {
        printf("error connection\n");
        return(NULL);
    }
    else
        printf("connected to host\n");
    return(co);
}

void get_ressources(virConnectPtr co)
{
    virNodeInfoPtr stat = NULL;

    if(!co)
        return ;
    char *ressources = virConnectGetCapabilities(co);
    (void)ressources;
    stat = malloc(sizeof(virNodeInfoPtr));
    if(!stat){perror("malloc stat"); return;}
    int y =  virNodeGetInfo(co, stat);
    if (y != -1)
    {
        printf("voici ressources %s\n", stat->model);

    }
}