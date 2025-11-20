#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "../include/electra/test_unit.h"
#include <string.h>
#include "../get_next_line/get_next_line_bonus.h"
#include <sys/inotify.h>
#include <pthread.h>
#include <signal.h>

typedef struct s_request
{
    char *h_host;
    char *h_port;
    char *h_methode;
    char *h_content_type;
    char *h_auth;
    char *h_content_len;
    char *connexion;
    char *body;

} t_request;

int exit_managment(int argc, char **argv)
{
    printf("bye bye\n");
    return(1);
}
#include <signal.h>

void print_event(struct inotify_event *event)
{
    if((event->mask & IN_CREATE) == IN_CREATE)
        printf("nouvelle object %s créer dans le repertoire\n", event->name);
    if(((event->mask & IN_DELETE) == IN_DELETE) ||((event->mask & IN_DELETE_SELF) == IN_DELETE_SELF))
        printf("object %s supprimmer du repertoire\n", event->name);
    if((event->mask & IN_MODIFY) == IN_MODIFY)
        printf("object %s modifier dans le repertoire\n", event->name);
    if((event->mask & IN_ACCESS) == IN_ACCESS)
        printf("object %s execution dans le repertoire\n", event->name);
    if(((event->mask & IN_MOVED_FROM) == IN_MOVED_FROM) ||((event->mask & IN_MOVED_TO) == IN_MOVED_TO))
        printf("object %s deplacé du repertoire\n", event->name);
}

#define EVENT_SIZE  ( sizeof (struct inotify_event) )
#define EVENT_BUF_LEN     ( 1024 * ( EVENT_SIZE + 16 ))

void *monitor_directory(void *data)
{
    int fd;
    char *path;
    char buffer[EVENT_BUF_LEN];
    struct inotify_event *event;

    path = (char *)data;
    fd = inotify_init();
    if(fd < 0) {perror("inotify error"); return(NULL);}
    int watch_d;
    memset(&event, 0 ,sizeof(struct inotify_event *));

    watch_d = inotify_add_watch(fd, path, IN_CREATE | IN_DELETE | IN_ATTRIB | IN_MOVED_TO | IN_MOVED_FROM | IN_MODIFY);
    if(watch_d < 0) {perror("inotify error"); return(NULL);}
    while (1)
    {
        int b = read(fd, buffer, EVENT_BUF_LEN);
        if(b <= 0){perror("read error"); return(NULL);}
        printf("read\n");
        if(b > 0)
        {
            event = (struct inotify_event *)buffer;
            print_event(event);
        }
    }

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

void get_directory_pool(char *root, char *path[100], int *idx, int depth)
{
    DIR *dir;
    struct dirent *d;

    if(!root || depth >= 4)
        return;
    dir = opendir(root);
    if(!dir){perror("Error open dir"); return;}
    while ((d = readdir(dir)) != NULL)
    {
        if(d->d_type == DT_DIR && d->d_name[0] != '.')
        {
            char *new_path = get_new_path(root, d->d_name);
            if(!new_path){printf("Error creation new path\n"); return;}
            //printf("new path: %s\n", new_path);

            //printf("add path %d\n", *idx);
            path[(*idx)] = ft_strdup(new_path);
            (*idx) += 1;
            get_directory_pool(new_path, path, idx, depth++);
            free(new_path);
        }
    }
    closedir(dir);
    return;
}


void *print_path(void *data)
{
    char *path;
    path = (char *)data;
    printf("thread for path %s\n", path);
    sleep(5);
    pthread_exit(NULL);
}

int clean_thread(pthread_t *thread, int idx)
{
    int i;

    i = 0;
    while (i < idx)
    {
        pthread_exit(&thread[i]);
        i++;
    }
    return(i);
}

int main(int argc, char **argv) 
{
    
    int i;
    int size;
    pthread_t thread[100];
    char *path[100];
    char *ptr;
    char *root = "/home/cc/gpu_lab/pleiades/electra/test_unit/models/";

    size = 0;
    memset(path, 0, sizeof(char *) * 100);
    get_directory_pool(root, path, &size, 0);
    if(!size)
    {
        printf("Error pool creation\n");
        return(1);
    }
    path[size] = NULL;
    memset(thread, 0, sizeof(pthread_t) * 100);
    i = 0;
    while (path[i])
    {
        if(pthread_create(&thread[i], NULL, monitor_directory, path[i]) < 0)
        {
            perror("%s\n");
            perror("Error thread creation");

        }
        i++;
    }
    i = 0;
    sleep(1);
    printf("I'am waiting for the trade\n");
    while (i < size)
    {
        pthread_join(thread[i],NULL);
        printf("Tread is back, nice %s\n", path[i]);

        i++;
    }
    i = 0;
    while(path[i])
    {
        free(path[i]);
        i++;
    }
    return 0;
}

//0.027836354449391365
//0.01694159209728241
//0.014803417026996613