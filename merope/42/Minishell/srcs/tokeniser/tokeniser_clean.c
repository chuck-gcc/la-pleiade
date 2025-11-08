#include "tokeniser.h"

void delete_list(void *content)
{
    t_token *token = (t_token *)content;

    if(token)
    {
        if(token->value)
        {
            free(token->value);
            token->value = NULL;
        }
        if(token->args)
        {
            char **tmp;

            tmp = token->args;
            while (*tmp)
            {
                free(*tmp);
                *tmp = NULL;
                tmp++;
            }
            free(token->args);
            token->args = NULL;
        }
        if(token->redir[0])
        {
            free(token->redir[0]);
            token->redir[0] = NULL;
        }
        if(token->redir[1])
        {
            free(token->redir[1]);
            token->redir[1] = NULL;
        }
        free(token);
        token = NULL;
    }
    
}