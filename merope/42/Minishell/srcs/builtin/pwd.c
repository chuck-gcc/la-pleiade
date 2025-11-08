#include "builtin.h"

#define BUFFER_SIZE 128

int ft_pwd(void)
{
    char path[BUFFER_SIZE];

    if(getcwd(path,BUFFER_SIZE) == NULL)
    {
        perror("Minishell: pwd");
        return(errno);
    }
    printf("%s\n", path);
    return(0);
}