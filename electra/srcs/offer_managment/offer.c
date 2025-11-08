#include "offer.h"

static char *create_offer_path(const char *str_1, const char *str_2 )
{
    char *path;
    int len_1, len_2;

    if(!str_1 || !str_2)
        return(NULL);
    len_1 = strlen(str_1);
    len_2 = strlen(str_2);
    path = malloc(sizeof(char) * (len_1 + len_2 + 1));
    if(!path){perror("Malloc"); return(NULL);}
    memcpy(path, str_1, len_1);
    memcpy(&path[len_1], str_2, len_2);
    path[len_2 + len_1] = '\0';
    return(path);
}

static t_offre *create_offer_node(struct dirent *dir_node)
{
    t_offre *node;
    xmlDoc  *doc = NULL;
    char *offer_path;
    
    if(!dir_node)
        return(NULL);
    node = malloc(sizeof(t_offre));
    if(!node){perror("malloc create_offer_node");return(NULL);}
    offer_path = create_offer_path(OFFERS_DIR_PATH, dir_node->d_name);
    if(!offer_path){free(node); return(NULL);}
    if((doc = xmlReadFile(offer_path, NULL,0)) == NULL)
    {
        printf("Error read xml file\n");
        free(offer_path);
        free(node);
        return(NULL);
    }
    printf("New XML doc was create: %s\n", dir_node->d_name);
    node->name = ft_strdup(dir_node->d_name);
    node->next = NULL;
    free(offer_path);
    xmlFreeDoc(doc);
    return(node);
}

static void offer_add_back(t_offre **offers_list, t_offre *offer)
{
    t_offre *ptr;

    if(!offers_list || !offer)
        return;
    if(!*offers_list)
    {
        *offers_list = offer;
        return;
    }
    ptr = *offers_list;
    while (ptr->next)
        ptr = ptr->next;
    ptr->next = offer;    
}

static void dislay_offer(t_offre **offre_list)
{
    t_offre *ptr;

    if(!offre_list)
        return;
    ptr = *offre_list;
    while(ptr)
    {
        printf("voici l'offre %s\n", ptr->name);
        ptr = ptr->next;
    }
}

t_offre **get_offer_list(void)
{
    FILE            *f;
    DIR             *dir;
    struct dirent   *dir_node;
    t_offre         **offers_list;
    t_offre         *offer;

    
    if((dir = opendir(OFFERS_DIR_OPEN))== NULL)
    {
        perror("open dir");
        return(NULL);
    }
    offers_list = malloc(sizeof(t_offre *));
    if(!offers_list){perror("malloc create_offer_list");closedir(dir);return(NULL);}
    while ((dir_node = readdir(dir)) != NULL)
    {
        if(strncmp(dir_node->d_name,".", strlen(dir_node->d_name)) == 0
        || strncmp(dir_node->d_name,"..", strlen(dir_node->d_name)) == 0)
            continue;
        offer = create_offer_node(dir_node);
        offer_add_back(offers_list, offer);
    }
    dislay_offer(offers_list);
    closedir(dir);
}