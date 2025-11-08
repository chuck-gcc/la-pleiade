#ifndef  OFFER_H
#define OFFER_H

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
#include <string.h>
 #include <dirent.h>
#include <sys/types.h>
#include <libxml/parser.h>
#include "../../../merope/42/libft/libft.h"

#define GPU_MAX 4
#define OFFERS_DIR_OPEN "srcs/offer_managment/offers"
#define OFFERS_DIR_PATH "srcs/offer_managment/offers/"

typedef struct s_offre
{
    char                *name;
    unsigned int        id;
    unsigned int        cpu;
    unsigned long       ram;
    unsigned int        gpu[GPU_MAX];
    unsigned int        vram;
    unsigned long       storage;
    char                *xml_path;
    struct s_offre      *next;
    
} t_offre;

t_offre **get_offer_list(void);

void dislay_offer(t_offre **offre_list);
int clean_list_offer(t_offre **offres);


#endif