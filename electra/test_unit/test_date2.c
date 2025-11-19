#include "../include/electra/test_unit.h"

#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include <dirent.h>
#define MENU_SIZE 4

struct termios oldt, newt;

void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

void die(const char *msg) {
    perror(msg);
    restore_terminal();
    exit(1);
}

char *get_new_path(char *current_dir, char *dest)
{
    int len_1;
    int len_2;
    char *new_path;

    if(!dest | !current_dir)
        return(NULL);
    len_1 = ft_strlen(current_dir);
    len_2 = ft_strlen(dest);
    new_path = malloc(sizeof(char) * (len_1 + len_2 + 2));
    if(!new_path){perror("error malloc new path"); return(NULL);}
    memcpy(new_path, current_dir, len_1);
    memcpy(&new_path[len_1], dest, len_2 );
    new_path[len_1 + len_2] = '/';
    new_path[len_1 + len_2 + 1] = '\0';
    return(new_path);
}

void reduce_path(char *current_path)
{
    int len;
    int count;
    if(!current_path)
        return;
    len = ft_strlen(current_path) - 1;
    count = 0;
    while (len >= 0)
    {
        if(current_path[len] == '/')
            count++;
        if(count == 2)
        {
            
            current_path[len + 1] = '\0';
            printf("start %d\n", len);
            return;
        }
        len--;
    }
}


int terminal_lock()
{
    if (tcgetattr(STDOUT_FILENO, &oldt) < 0) die("tcgetattr");
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    if (tcsetattr(STDOUT_FILENO, TCSANOW, &newt) < 0)
        die("tcsetattr");
    atexit(restore_terminal);
    return(0);
}

int main(int argc, char *argv[]) {

    (void)argc;
    char *current_dir;
    char *next_dir;
    struct dirent *folder;
    DIR *dir, *dir2;
    int i, c , idx;
    
    next_dir = NULL;
    current_dir = ft_strdup(getenv("MODEL"));
    if(!current_dir){printf("No env variable\n");return(1);}
    terminal_lock();
    i = 1;
    system("clear");
    while (1) {

        idx = 0;
        dir = opendir(current_dir);
        if(dir)
            dir2 = dir;
        else
            dir = dir2;
        while((folder = readdir(dir)) != NULL)
        {
            if(idx == 0)
                printf("current dir: %s\n",current_dir);
            else if(idx != i)
                printf("%s\n",folder->d_name);
            else
            {
                if(next_dir)
                    free(next_dir);
                next_dir = ft_strdup(folder->d_name);
                printf("\e[41m""%s""\e[0m""\n",folder->d_name);
            }
            idx++;
        }
        c = 0;
        int n = read(STDIN_FILENO, &c, 1* sizeof(int));
        if (n > 0) {
            if(c == 4348699)
            {
                system("clear");

                i + 1 >= idx  ? i = 1 : i++;
            }
            else if(c == 4283163)
            {
                system("clear");
                
                i - 1 < 1 ? i = idx - 1 : i--;
            }
            else if(c == 10)
            {
                char *new_path;
                if(i >= idx)
                {
                    reduce_path(current_dir);
                }
                else
                {

                    new_path = get_new_path(current_dir, next_dir);
                }
                if(!new_path){printf("Erreur get new path\n"); return(1);}
                system("clear");
                free(current_dir);
                current_dir = new_path;
                i = 1;
            }
            else if(c == 113)
            {
                printf("EXIT\n");
                closedir(dir);
                free(next_dir);
                free(current_dir);

                restore_terminal();
                return(0);    
            }
            else
                printf("voici %d\n",c);
        }
        else
            perror("erruer \n");
        closedir(dir);
    }
    
    return 0;
}


