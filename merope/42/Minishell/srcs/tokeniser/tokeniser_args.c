#include "tokeniser.h"

static char *ft_get_env(char *var, char **envp)
{
    int i;
    char *var_env;
    char *value;

    i = 0;
    while (envp[i])
    {
        var_env = ft_substr(envp[i], 0, ft_index_of_c(envp[i], '='));
        if(var_env)
        {
            if(ft_strncmp(var, var_env, ft_strlen_longest(var, var_env)) == 0)
            {
                value = &envp[i][ft_index_of_c(envp[i], '=') + 1];
                free(var_env);
                return(value);
            }
        }
        free(var_env);
        i++;
    }
    return(NULL);
}

static int count_args(char **input)
{
    int i;

    if(!input)
        return(-1);
    i = 0;
    while (input[i] && get_token_type(input[i]) != PIPE)
        i++;
    return(i);
}

int ft_expend_var(t_token *token, char **envp, int status_code)
{
    int i;
    int idx_in_var;
    char *new_var;
    char *var;
    int j; 
    int k;

    if(!token || !token->args)
        return(1);
    i = 0;
    while (token->args[i])
    {
        idx_in_var = ft_index_of_c(token->args[i], '$');
        if(idx_in_var == 0)
        {
            if(ft_strncmp(token->args[i], "$?", ft_strlen_longest(token->args[i], "$?")) == 0)
            {
                free(token->args[i]);
                token->args[i] = ft_strdup(ft_itoa(status_code));
            }
            else
            {
                var = ft_get_env(&token->args[i][1], envp);
                free(token->args[i]);
                if(!var)
                    token->args[i] = ft_strdup(" ");
                else
                    token->args[i] = ft_strdup(var);
            }
        }
        else if (idx_in_var > 0)
        {

            j = 0;
            k = 0;
            var = ft_get_env(&token->args[i][idx_in_var + 1], envp);
            if(var)
            {

                new_var = malloc(sizeof(char) * (idx_in_var + ft_strlen(var) + 1));
                if(!new_var)
                    return(-1);
                while (j < idx_in_var)
                {
                    new_var[j] = token->args[i][j];
                    j++;
                }
                while (var[k])
                {
                    new_var[j] = var[k];
                    j++;
                    k++;
                }
                new_var[j] = '\0';
                token->args[i] = new_var;
            }
        }
        i++;
    }
    return(0);
}

int get_args(t_list *node, char **input, char **envp, int status_code)
{
    int i;
    int idx;
    int args_count;
    char **args;

    i = 0;
    idx = 0;
    args_count = 0;
    if(*input)
        args_count = count_args(input) + 1;
    if(args_count)
    {
        args = malloc(sizeof(char *) * (args_count) + 1);
        if(!args)
        {
            perror("get args");
            return(-1);
        }
        args[i++] = ft_strdup(((t_token *)node->content)->value);
        while (get_token_type(input[idx]) != PIPE && !is_redir(input[idx]))
            args[i++] = ft_strdup(input[idx++]);
        args[i] = NULL;
        ((t_token *)node->content)->args = args;
        ft_expend_var(((t_token *)node->content), envp, status_code);
        return(idx);
    }
    return(0);
}
