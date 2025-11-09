#include "offer.h"



void binary_resa_tree(t_resa *tree)
{

    t_offre *content;

    if(!tree)
        return;

    
    binary_resa_tree(tree->left);
    content = tree->content;
    printf("resa id: %d --> : %s start: %ld end %ld\n",content->id, content->name, tree->interval[0], tree->interval[1]);
    binary_resa_tree(tree->right);

}


t_resa *ask_a_resa(t_offre *offre, time_t start, time_t end)
{
    t_resa *resa;

    resa = malloc(sizeof(t_resa));
    if(!resa){perror("malloc resa"); return(NULL);}
    resa->content = offre;
    resa->interval[0] = start;
    resa->interval[1] = end;
    resa->left = NULL;
    resa->right = NULL;
    return(resa);
}


int test_time(t_offre **list)
{
    time_t start, end;
    t_resa *resa_tree;
    struct tm  *t;

    start = time(NULL);
    t =  localtime(&start);
    printf("millisecond since January 1, 1970: %ld\n", start);
    printf("Nous somme le %d/%d/%d\n", t->tm_mday, t->tm_mon + 1, t->tm_year + 1900);
    printf("Il est %dH%d\n",t->tm_hour, t->tm_min);
    unsigned long dif = (unsigned long)difftime(end, start);
    printf("voici la diff %ld \n", dif);

    resa_tree = NULL;

    t_resa *resa;
    
    resa = ask_a_resa(*list, start, start + 1);




    binary_resa_tree(resa);
}