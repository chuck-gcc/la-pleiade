
#include "main.h"
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>


char **dupplicate_env(char **old_env)
{
    char **env;
    int i;
    int old_env_len;

    old_env_len = ft_get_split_len(old_env);
    if(old_env_len <= 0)
        return(NULL);
    env = malloc(sizeof(char *) * (old_env_len + 1));
    if(!env)
        return(NULL);
    i = 0;
    while (old_env[i])
    {
        env[i] = ft_strdup(old_env[i]);
        if(!(env[i]))
            return(ft_split_clean(&env));
        i++; 
    }
    env[i] = NULL;
    return(env);
}

static int process_user_input(char *str, t_env *self_env, int status_code)
{
    t_list  **tokens_lst;
    int     status;


    tokens_lst = calloc(sizeof(t_list *) , 1);
    if(!tokens_lst)
        return(1);
   
    if(!get_token_list(str, tokens_lst, *self_env->env, status_code))
    {
        printf("Error token list\n");
        ft_lstclear(tokens_lst, delete_list);
        return(1);
    }

    printf("\n");
    ft_lstiter(*tokens_lst, display_args_of_cmd);

    t_token **ast = malloc(sizeof(t_token *));
    if(!ast)
    {
        ft_lstclear(tokens_lst, delete_list);
        return(1);
    }
    *ast = NULL;
    generate_ast(*tokens_lst, ast);
    if(!*ast)
    {
        ft_lstclear(tokens_lst, delete_list);
        return(1);
    }

    display_binary_tree(NULL,*ast,0);
    printf("\n");
    int saved =  dup(STDOUT_FILENO);

    status = execute_ast(*ast, self_env, saved);
    //int r = execute_heredoc(*ast, "n",*envp);
    // important know
    
    ft_lstclear(tokens_lst, delete_list);
    free(ast);
    return(status);
}




int run_minishell(t_env *self_env)
{
    char *input;
    int status_code;

    
    status_code = -1;
    if(!self_env || !self_env->env)
    {
        perror("Error Env\n");
        return(1);
    }
    while (1)
    {
        input = readline("mini michel: ");
        if(input)
        {
            if(*input)
                add_history(input);
            // if(!ft_strncmp(input,"exit", ft_strlen_longest(input,"exit")))
            // {
            //     free(input);
            //     clear_history();
            //     return(1);
            // }
            //printf("voici le status %s\n", input);
            status_code = process_user_input(input, self_env, status_code);
            //printf("STATUS COMMANDE %d\n\n", status);
            free(input);
            rl_on_new_line();
        }
    }
    return(0);
}

int main(int argc, char **argv, char **envp)
{

    t_env *self_env;
    
    self_env = init_env(envp);
    if(!self_env)
        printf("error\n");
    run_minishell(self_env);
    return(0);
}