#include "ast.h"

void display_binary_tree(t_token *parent, t_token *actual, int deriv)
{
    (void)parent;

    if(actual == NULL)
    {
        return;
    }
    
    parent = actual;

    printf("%-10s | %-15s | precedence: %-3d | asso: %-3d\n",actual->value, print_token_type(actual->type),actual->precedence, actual->asso);

    display_binary_tree(actual,actual->left, deriv);
    

    if(deriv)
    {
        if(!actual->right)
    
            printf("\033[0;31m" "\ngo in right of :%-5s: Nothing\n\n" "\033[0m", parent->value);
        else
            printf("\033[0;32m""go in right of :%-10s\n\n""\033[0m", parent->value);
    }

    display_binary_tree(actual, actual->right, deriv);
}

int generate_ast(t_list *token_list, t_token **ast_root)
{
    t_token *token;
    
    if(!token_list)
        return(1);

    token =  ((t_token *)token_list->content);

    if(!*ast_root)
    {
        *ast_root = token;
        //printf("new root\n");
        generate_ast(token_list->next, ast_root);
    }
    else
    {
        if(token->type == PIPE)
        {
            t_token *tmp = *ast_root;
            *ast_root = token;
            (*ast_root)->left = tmp;
            //printf("new root %s\n", token->value);
            generate_ast(token_list->next, ast_root);
        }
        else
        {
            if(!(*ast_root)->left)
                (*ast_root)->left = token;
            else
                (*ast_root)->right = token;
            generate_ast(token_list->next, ast_root);
        }
    }
    return 1;
}
