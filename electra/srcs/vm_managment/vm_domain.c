#include "vm.h"

int list_actif_domain(virConnect *conn)
{
    virDomainPtr *doms = NULL;
    int i;

    if(conn)
    {
        int r = virConnectListAllDomains(conn, &doms, (VIR_CONNECT_LIST_DOMAINS_SHUTOFF | VIR_CONNECT_LIST_DOMAINS_RUNNING));
        if(r > 0)
        {
            printf("Nombre de machine sur host: %d\n",r);
            i = 0;
            while (i < r)
            {
                if(virDomainIsActive(doms[i]))
                    printf("the machine %s is active\n", virDomainGetName(doms[i]));
                else
                    printf("the machine %s is inativ\n", virDomainGetName(doms[i]));
                virDomainFree(doms[i]);
                i++;
            }
        }
        free(doms);
        return(0);
    }
    return(1);
}


int create_vm(virConnect *conn)
{
    int fd;
    virDomainPtr    new_vm;
    long xml_len;
    char *xml;

    xml = get_xml("VM_catalog/template_test.xml");
    new_vm =  virDomainDefineXML(conn,xml);
    if(new_vm == NULL)
    {
        printf("Erreur define xml\n");
        virConnectClose(conn);
        return(1);
    }
    else
        printf("new vm initialised\n");
    virDomainCreate(new_vm);
    free(xml);
    virConnectClose(conn);
    return(0);
}