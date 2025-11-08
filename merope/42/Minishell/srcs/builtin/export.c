#include "builtin.h"



/*
    A  word  consisting  only  of alphanumeric characters and under‐
    scores, and beginning with an alphabetic character or an  under‐
    score.  Also referred to as an identifier
*/
int is_valide_variable(char *var)
{
    char *ptr;
    int equal;

    ptr = var;
    equal = 0;
    if(!ft_isalpha(*ptr))
        return(-1);
    while(*ptr)
    {
        if(*ptr == '=')
        {
            equal = 1;
            ptr++;
        }
        if(!ft_isalnum(*ptr) && equal == 0)
            return(-1);
        ptr++;
    }

    return(0);
}


size_t count_valide_variable(char **vars, int mode)
{
    char **ptr;
    size_t i;

    ptr = vars;
    i = 0;
    while (*ptr)
    {
        if(is_valide_variable(*ptr) == 0)
            i++;
        else
        {
            if(mode == EXPORT)
                printf("minishell: export: `%s': not a valid identifier\n", *ptr);
            else if(mode == UNSET)
                printf("minishell: unset: `%s': not a valid identifier\n", *ptr);
        }
        ptr++;
    }
    return(i);
}

int is_on_env(char *var, char **env, int len)
{
    int i;
    int idx_env; 
    int idx_var; 
    char *sub_var; 
    char *sub_env;
    
    i = 0;
    
    idx_var = ft_index_of_c(var,'=');
    sub_var = ft_substr(var, 0, idx_var);
    while (i < len)
    {
        idx_env =  ft_index_of_c(env[i],'=');
        sub_env = ft_substr(env[i], 0, idx_env);
        if(ft_strncmp(sub_env, sub_var, ft_strlen_longest(sub_var, sub_env)) == 0)
        {
            free(sub_env);
            free(sub_var);
            return(i);
        }
        i++;
    }
    free(sub_env);
    free(sub_var);
    return(-1);
}

char **get_new_env(char **envp, char **args)
{
    char **new_env;
    int on_env;
    size_t len_env;
    size_t valide_var;
    size_t  j;
    size_t  i;

    len_env = ft_get_split_len(envp);
    valide_var = count_valide_variable(args, EXPORT);
    if(valide_var == 0)
        return(NULL);
    new_env = malloc(sizeof(char *) * (len_env + valide_var + 1));
    if(!new_env)
        return(NULL);
    ft_memcpy(new_env, envp, sizeof(char *) * len_env);
    i = 0;
    while (envp[i])
    {
        new_env[i] = ft_strdup(envp[i]);
        i++;
    }
    j = 0;
    while (args[j])
    {
        if(is_valide_variable(args[j]) != -1)
        {
            on_env = is_on_env(args[j], new_env, len_env);
            if(on_env >= 0)
            {
                free(new_env[on_env]);
                new_env[on_env] = ft_strdup(args[j]);
            }
            else
                new_env[i++] = ft_strdup(args[j]);
        }
        j++;
    }
    new_env[i] = NULL;
    return(new_env);
}


int ft_export(t_env *env, t_token *token)
{
    char    **new_env;

    if( !env || !env->env || !(*env->env) || !token)
    {
        printf("Error env :\n");
        return(-1);
    }
    if(!token->args)
    {
        env->display_export_env(env);
        return(0);
    }
    new_env = get_new_env(*env->env, &token->args[1]);
    if(!new_env)
    {
        printf("error creation new env\n");
        return(1);
    }
    else
    {
        if(env->swap_env(env,new_env) == 1)
        {
            printf("Error swap env\n");
            return(1);
        }
    }
    return(0);
}