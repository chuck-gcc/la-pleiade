#include "tokeniser.h"


int ft_is_builtin(char *str)
{
    if
    (   !ft_strncmp(str, "echo", ft_strlen_longest(str, "echo")) ||
        !ft_strncmp(str, "cd", ft_strlen_longest(str, "cd")) ||
        !ft_strncmp(str, "pwd", ft_strlen_longest(str, "pwd")) ||
        !ft_strncmp(str, "export", ft_strlen_longest(str, "export")) ||
        !ft_strncmp(str, "unset", ft_strlen_longest(str, "unset")) ||
        !ft_strncmp(str, "env", ft_strlen_longest(str, "env")) ||
        !ft_strncmp(str, "exit", ft_strlen_longest(str, "exit")) 
    )
        return(1);
    else
        return(0);
}


//using opendir and readdir for check if value is commande
// looping inside th repository
int ft_is_commande(char *str)
{
    char *path;

    path = get_path(str);
    if(!path)
        return(0);
    free(path);
    return(1);
}

int get_token_type(char *str)
{
    if(!str)
        return(0);
    if(ft_is_builtin(str))
    {
        return(BUILTIN);
    }
    if(ft_is_commande(str))
        return(CMD);
    if(!ft_strncmp(str, "|", ft_strlen_longest(str, "|")))
        return(PIPE);
    if(!ft_strncmp(str, ">", ft_strlen_longest(str, ">")))
        return(REDIR_RIGHT);
    if(!ft_strncmp(str, "<", ft_strlen_longest(str, "<")))
        return(REDIR_LEFT);
    if(!ft_strncmp(str, ">>", ft_strlen_longest(str, ">>")))
        return(REDIR_APPEND);
    if(!ft_strncmp(str, "<<", ft_strlen_longest(str, "<<")))
        return(DELIM);
    return(WORD);
}

int get_precedence(int token_type)
{
    if(token_type == PIPE)
        return(3);
    if(token_type == CMD)
        return(2);
    else
        return(1);
}

int get_asso(int token_type)
{
    if(token_type == PIPE)
        return(3);
    if(token_type == CMD)
        return(2);
    else
        return(1);
}



