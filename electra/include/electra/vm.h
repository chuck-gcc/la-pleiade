#ifndef  VM_MANAGMENT_H
#define VM_MANAGMENT_H

#include <stdlib.h>
#include <assert.h>
#include <libvirt/libvirt.h>
#include <libvirt/libvirt-domain.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>


typedef struct s_orchestrator
{
    virConnectPtr co;
    virNodeInfoPtr host_info;
    
} t_orchestrator;


virConnectPtr host_connection(void);
void get_ressources(virConnectPtr co);

char *get_xml(const char *path);
int create_vm(virConnect *conn);


#endif