#include "builtin.h"

int check_variable(char **var)
{
    int i;

    i = 0;
    if(!var)
        return(0);
    while (var[i])
    {
        if(ft_index_of_c(var[i], '=') == -1)
        {
            printf("env: '%s': No such file or directory\n", var[i]);
            return(-1);
        }
        i++;
    }
    return(0);
}

int ft_env(t_token *token, t_env *env)
{
    char **ptr;
    int i;

    if(!token || !env || !env->env)
        return(-1);
    if(!token->args)
    {
        env->print_env(env);
        return(0);
    }
    i = 1;
    if(check_variable(&token->args[i]) == -1)
        return(1);
    ptr = *env->env;
    while (*ptr)
    {
        printf("%s\n", *ptr);
        ptr++;
    }
    while (token->args[i])
    {
        printf("%s\n", token->args[i]);
        i++;
    }
    return(0);
}