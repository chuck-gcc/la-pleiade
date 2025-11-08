#include "ast.h"

void destroy_token(t_token **tk)
{
    t_token *token; 

    token = *tk;
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
        free(*tk);
        *tk = NULL;
    }
}

int read_here(int *in, int *out, char *delim, int saved)
{
    char    *str;
    int     w;
    int s;

    //important to commante every stout write. for cat << t | wc -l, every line put on stdout in cat children are count in wc.
    //printf("voici le delim %s\n",delim);
    if((in == NULL ) | (out == NULL))
        return(1);
    w = 0;
    if(w == -1){perror("heredoc write 1"); return(errno);}
    s = dup(STDOUT_FILENO);
    (void)s;
    dup2(saved, STDOUT_FILENO);
    while ((str = readline("heredoc> ")) != NULL)
    {
        if(str == NULL)
            return(1);
        if(ft_strncmp(str, delim, ft_strlen(str)) == 0)
        {
            dup2(s, STDOUT_FILENO);
            return(0);
        }
        else
        {
            w = write(*out, str, ft_strlen(str));
            if(w == -1){perror("heredoc write 1"); return(errno);}
            w = write(*out, "\n", ft_strlen("\n"));
            if(w == -1){perror("heredoc write 2"); return(errno);}
        }
    }

    return(1);
}

int execute_redirection(t_token *token, int *tube, int saved)
{
    int r;

    r = 1;
    if(is_redir(token->redir[0]) == DELIM)
        r = read_here(&tube[0], &tube[1], token->redir[1], saved);
    return(r);

}

int execute_commande(t_token *token, char *path, t_env *self_env, int saved)
{
    int status;
    int tube[2];
    status = 0;
    
    
    if(pipe(tube) == -1) { perror("pipe"); return (-1);}
    
    pid_t f1 = fork();

    if((f1 == -1) ) { perror("fork"); return (-1);}
    if(f1 == 0)
    {
        close(tube[0]);
        if(token->redir_type != -1)
        {
            execute_redirection(token, tube, saved);
            close(tube[1]);
            exit(0);
        }
        close(tube[1]);
        //exit(0);

        
    }
    waitpid(f1,&status, 0);
    if(WIFEXITED(status))
    {
        pid_t f2 = fork();
        if((f2 == -1) ) { perror("fork"); return (-1);}
        if(f2 == 0)
        {
            close(tube[1]);
            dup2(tube[0], STDIN_FILENO);
            close(tube[0]);

            execve(path, token->args, *self_env->env);
            perror("Execution error");
            exit(errno); 
        }
        close(tube[0]);
        close(tube[1]);
        waitpid(f2,&status, 0);
        if(WIFEXITED(status))
            return(WEXITSTATUS(status));
        else
        {
            printf("Error %d\n", WEXITSTATUS(status));
            return(WEXITSTATUS(status));
        }
    }
    else
    {
        printf("Error %d\n", WEXITSTATUS(status));
        close(tube[0]);
        close(tube[1]);
        return(WEXITSTATUS(status));
    }
    return(WEXITSTATUS(status));
}


int execute_builtin(t_token *token, t_env *env)
{
    if(!token)
        return(1);
    if(ft_strncmp(token->value, "cd", ft_strlen(token->value)) == 0)
        return(ft_cd(token));
    if(ft_strncmp(token->value, "env", ft_strlen(token->value)) == 0)
        return(ft_env(token, env));
    if(ft_strncmp(token->value, "pwd", ft_strlen(token->value)) == 0)
        return(ft_pwd());
    if(ft_strncmp(token->value, "echo", ft_strlen(token->value)) == 0)
        return(ft_echo(token));
    if(ft_strncmp(token->value, "export", ft_strlen(token->value)) == 0)
        return(ft_export(env,token));
    if(ft_strncmp(token->value, "unset", ft_strlen(token->value)) == 0)
        return(ft_unset(env,token));
    if(ft_strncmp(token->value, "exit", ft_strlen(token->value)) == 0)
        ft_exit();
    return(1);
}

int      execute_ast(t_token *ast, t_env *self_env, int saved)
{
    int status;
    int r;
    int tube[2];


    if(!ast)
        return(0);

    status = 0;
    if(pipe(tube) == -1) {perror("fork"); return (-1);}
    
    if(ast->type == PIPE)
    {
        
        pid_t f1 = fork();
        if(f1 == -1) { perror("fork"); return (-1);}
        if(f1 == 0)
        {   

            close(tube[0]);
            dup2(tube[1], STDOUT_FILENO);
            execute_ast(ast->left, self_env, saved);
            
            dup2(STDOUT_FILENO, saved);
            exit(0);
        }
        waitpid(f1,&status, 0);
        
        pid_t f2 = fork();
        int status2 = 0;
        if(f2 == -1) { perror("fork"); return (-1);}
        if(f2 == 0)
        {

            close(tube[1]);
            dup2(tube[0], STDIN_FILENO);
            close(tube[0]);
            exit(execute_ast(ast->right, self_env, saved));
        }

        close(tube[0]);
        close(tube[1]);
        waitpid(f2,&status2, 0);

        return (status);
    }
    if(ast->type == BUILTIN)
    {
        r = execute_builtin(ast, self_env);
        return(r);
    }
    else if(ast->type == CMD)
    {
        char *path = get_path(ast->value);
        if(!path)
            return(-1);
        r = execute_commande(ast, path, self_env, saved);
        return(r);
    }
    return (0);
}