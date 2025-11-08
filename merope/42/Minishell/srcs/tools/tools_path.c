#include "tools.h"

char *get_os(void)
{
    int tube[2];
    char os[100];
    int status;
    int i;
    i = 0;
    if(pipe(tube) == -1) {perror("pipe"); return (NULL);}
    pid_t f;

    f = fork();
    if(f == -1) {perror("fork"); return(NULL);}
    if(f == 0)
    {
        close(tube[0]);
        dup2(tube[1], STDOUT_FILENO);
        close(tube[1]);
        char *argv[] = { "uname", NULL};
        if(execve("/usr/bin/uname", argv, NULL) == -1){perror("excve");exit(errno);}
    }
    else
    {
        close(tube[1]);
        waitpid(f,&status,0);
        if((i = read(tube[0], os, 100)) == -1)
            printf("Error wirte\n");
        os[i - 1] = '\0';
        close(tube[0]);
    }
    return(ft_strdup(os));
}

char *get_path_for_mac(char *cmd)
{
    
    DIR     *folder;
    struct  dirent *s;

    char *path[3];
    int i;

    path[0] = "/bin/";
    path[1] = "/usr/bin/";
    path[2] = NULL;
    i = 0;
    while (path[i])
    {
        folder = opendir(path[i]);
        while(folder)
        {
            s = readdir(folder);
            if(!s)
                break;
            if(ft_strncmp(cmd,s->d_name, ft_strlen_longest(cmd, s->d_name)) == 0)
            {
                closedir(folder);
                folder = NULL;
                return(ft_strdup(path[i]));
            }
        }
        closedir(folder);
        folder = NULL;
        i++;
    }
    return(NULL);
}

char *get_path_for_linux(char *cmd)
{
    DIR     *folder;
    struct  dirent *s;

    folder = opendir("/usr/bin/");
    while(folder)
    {
        s = readdir(folder);
        if(!s)
            break;
        if(ft_strncmp(cmd,s->d_name, ft_strlen_longest(cmd, s->d_name)) == 0)
        {
            closedir(folder);
            folder = NULL;
            return(ft_strdup("/usr/bin/"));
        }
    }
    closedir(folder);
    folder = NULL;
    return(NULL);
}

char *get_base_path(char *str)
{
    char *final_path;

    if(!str)
        return(NULL);
    #ifdef __linux__ 
        final_path = get_path_for_linux(str);
        if(!final_path)
            return(NULL);
    #else
        final_path = get_path_for_mac(str);
        if(!final_path)
            return(NULL);
    #endif
    return(final_path);
}

char *get_path(char *str)
{
    char *path;
    char *base;
    int len_1;
    int len_2;
    int i;
    int j;

    if(!str)
        return(NULL);
    base = get_base_path(str);
    if(!base)
        return(NULL);
    len_1 = ft_strlen(base);
    len_2 = ft_strlen(str);
    i = 0;
    j = 0;
    path = malloc(sizeof(char *) * (len_1 + len_2 ) + 1);
    if(!path)
    {
        free(base);
        return(NULL);
    }
    while (base[i])
        path[j++] = base[i++];
    i = 0;
    free(base);
    while (str[i])
        path[j++] = str[i++];
    path[j] = '\0';
    return(path);
}