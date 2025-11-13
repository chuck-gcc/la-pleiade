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
 #include <time.h>
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
    unsigned int        resa_count;
    struct s_offre      *next;
    
} t_offre;


typedef struct s_resa
{

    time_t          start;
    time_t          end;
    int             offer_id;

    struct s_resa *left;
    struct s_resa *right;

} t_resa;



int test_time(t_offre **list);

t_offre **get_offer_list(void);
void    dislay_offer_list(t_offre **offre_list);
int     clean_list_offer(t_offre **offres);
t_resa *iso_date_to_resa(const char *iso_date, char *id, char *time);
void    display_resa(t_resa *root);
int add_resa(t_resa *root, t_resa *resa);

#endif