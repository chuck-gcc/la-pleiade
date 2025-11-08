
#include "builtin.h"

void ft_exit()
{
    pid_t pid;

    pid = getpid();

    //int s = kill(pid,0);
    printf("nous somme dans le pid %d voici le retour de s \n", pid);

    // if(s == -1)
    // {
    //     perror("KILL ");
    // }

}