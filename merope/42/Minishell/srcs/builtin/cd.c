#include "builtin.h"

int ft_cd(t_token *token)
{
    char *path;
    if(!token)
        return(1);
    if(!token->args)
    {
        path = getenv("HOME");
        if(chdir(path) != 0)
        {
            perror("Minishell: cd");
            return(errno);
        }
        return(0);
    }
    if(ft_get_split_len(token->args) > 2)
    {
        printf("Minishell: cd: too many arguments\n");
        return(1);
    }
    if(!token->args)
    {
        printf("here\n");
    }
    if(ft_strncmp(token->args[1],"~",ft_strlen(token->args[1])) == 0 || !token->args[1])
        path = getenv("HOME");
    else
        path = token->args[1];
    if(chdir(path) != 0)
    {
        perror("Minishell: cd");
        return(errno);
    }
    return(0);
}
