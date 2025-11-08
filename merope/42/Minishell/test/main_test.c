#include "../srcs/main.h"
#include <errno.h>
#include <assert.h>


int get_ssredir_option(int redir_type)
{
    if(redir_type == REDIR_RIGHT)
        return(O_WRONLY);
    if(redir_type == REDIR_APPEND)
    {
        printf("voici %d\n", O_APPEND);
        return(O_WRONLY | O_APPEND);
    }
    if((redir_type == REDIR_LEFT) | (redir_type == DELIM))
        return(O_RDONLY);
    return(-1);
}

int read_here(int *in, int *out, char *delim)
{
    char    *str;
    int     w;

    if(in == NULL  | out == NULL)
        return(1);
    w = close(*in);
    if(w == -1){perror("heredoc write 1"); return(errno);}
    while ((str = readline("heredoc> ")) != NULL)
    {
        if(str == NULL)
            return(1);
        if(ft_strncmp(str, "t", ft_strlen(str) - 1) == 0)
            return(0);
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

int main(int argc, char **argv,char **envp)
{
    printf("voici la fonction test main\n");
    int tube[2];
    pid_t pid;
    int status;

    if(pipe(tube) == -1){perror("tube"); return (errno);}
    pid = fork();
    if(pid == -1){perror("test"); return(1);}
    if (pid == 0)
    {
        read_here(&tube[0],&tube[1], "t");
        exit(0);
    }
    else
    {   
        waitpid(pid, &status, 0);
        if(WIFEXITED(status))
        {
            char *arg[] = {"cat",NULL};
            char *cmd = "/bin/cat";

            close(tube[1]);
            dup2(tube[0],STDIN_FILENO);

            close(tube[0]);

            int e = execve(cmd, arg, envp);
            if(e == -1){perror("excve test"); return(1);}

        }

    }
    return (0);
}