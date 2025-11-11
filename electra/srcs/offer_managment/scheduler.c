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
    binary_resa_tree(tree->right);

}

long date_to_ms(int day, int mount, int year, int hour, int min)
{

    

}


void printf_date(time_t start, time_t end)
{
    struct tm *date_s;
    struct tm *date_e;

    date_s = localtime(&start);
    date_e = localtime(&end);

    printf("START ---> %d/%d/%d %d:%d:%d\n", date_s->tm_mday, date_s->tm_mon, 1900 + date_s->tm_year, date_s->tm_hour, date_s->tm_min,date_s->tm_sec);
    printf("END -----> %d/%d/%d %d:%d:%d\n", date_e->tm_mday, date_e->tm_mon, 1900 + date_e->tm_year, date_e->tm_hour, date_e->tm_min,date_e->tm_sec);
    printf("\n\n");
}

int add_resa(t_resa *root, t_resa *resa)
{
    t_resa *tmp;

    if(!root || !resa)
        return(1);
    if(!root->left)
    {
        root->left = resa;
        return(0);
    }
    if(!root->right)
    {
        root->right = resa;
        return(0);
    }
    else
    {
        tmp = root;
        root = resa;
        resa->left = tmp;
        return(0);
    }
    return(1);
}

void display_resa(t_resa *root)
{
    if(!root)
        return;
    display_resa(root->left);

    printf_date(root->start, root->end);

    display_resa(root->right);
}

int create_resa(struct tm *date, int hour, int id, t_resa **resa_tree)
{

    t_resa *resa; 

    if(!*resa_tree)
        printf("l'arbre de reservation est NULL\n");
    
    resa = malloc(sizeof(t_resa));
    if(!resa){perror("malloc"); return(1);}

    resa->start = mktime(date);
    resa->end = resa->start + 3600  ;
    resa->offer_id = id;
    resa->right = NULL;
    resa->left = NULL;
    
    if(!*resa_tree)
    {
        *resa_tree = resa;
        return(0);
    }
    
    int r = add_resa(*resa_tree, resa);
    return(r);
}