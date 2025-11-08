#include "tokeniser.h"

char *print_token_type(int token_type)
{
    if(token_type == CMD)
        return("COMMANDE");
    if(token_type == WORD)
        return("WORD");
    if(token_type == BUILTIN)
        return("BUILTIN");
    if(token_type == PIPE)
        return("PIPE");
    if(token_type == REDIR_LEFT)
        return("REDIR LEFT");
    if(token_type == REDIR_RIGHT)
        return("REDIR RIGHT");
    if(token_type == REDIR_APPEND)
        return("REDIR APPEND");
    if(token_type == DELIM)
        return("DELIM");
    return(NULL);
}

void display_content_lst(void *liste)
{
    t_token *token;
    char *value;
    
    token = (t_token *)liste;
    value = token->value;

    printf("%-10s | %-15s | precedence: %-3d | asso: %-3d\n",value,
    print_token_type(token->type),
    token->precedence,
    token->asso);
}

void display_args_of_cmd(void *liste)
{
    t_token *token;
    token = (t_token *)liste;
    display_arg_of_cmd(token);
}
void display_arg_of_cmd(t_token *token)
{
    if(!token)
        return;
    if(token->type == CMD || token->type == BUILTIN  || token->type == PIPE)
    {
        printf("Commande: ");
        printf("%s\n", token->value);
        printf("Args: ");
        if(!token->args)
            printf("\033[31m%s\033[0m\n", "No arguments");
        else
        {
            int i = 0;
            while (token->args && token->args[i])
            {
                printf("[%s]", token->args[i]);
                i++;
            }
            printf("\n");
        }
        printf("Redir type: %d\n",token->redir_type);
        printf("Redir: ");
        if(token->redir_type >= 0)
        {
            
            printf("[%s]", token->redir[0]);
            printf("[%s]", token->redir[1]);
            printf("\n");
        }
        else
            printf("\033[31mNo redirection\033[0m\n");
    }
    printf("\n");
}
