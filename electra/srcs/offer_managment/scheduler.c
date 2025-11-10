#include "offer.h"

#define HourToMs(hour) (hour * (60 * (60 * 1000)))


void add_resa_node(t_resa *tree, t_resa *node)
{

    if(!tree)
        return;

    if(tree->left != NULL && tree->right != NULL)
    {
        node->left = tree;
        return;
    }
    add_resa_node(tree->left, node);
    add_resa_node(tree->right, node);


}

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
    t_resa *resa1;
    t_resa *resa2;
    t_resa *resa3;
    t_resa *resa4;
    
    resa = ask_a_resa(*list, start, start + 1);
    resa1 = ask_a_resa((*list)->next, start + 1000, start + 2000);
    resa2 = ask_a_resa((*list)->next->next, start + 4000, start + 6000);
    resa3 = ask_a_resa((*list)->next->next->next, start + 4000, start + 6000);
    resa4 = ask_a_resa((*list)->next, start + 6000, start + 6000);

    resa->left = resa1;
    resa->right = resa2;

    binary_resa_tree(resa);
    printf("new start\n");
    resa3->left = resa;
    resa3->right = resa4;
    binary_resa_tree(resa3);
}