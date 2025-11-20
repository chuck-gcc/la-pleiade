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


typedef struct s_log_file
{
    int idx;
    char *path;
} t_log_file;



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
void *monitor_file(void *data)
{
    int fd;
    char *path;
    char buffer[EVENT_BUF_LEN];
    struct inotify_event *event;

    path = (char *)data;
    printf("voici file %s\n", path);
    fd = inotify_init();
    if(fd < 0) {perror("inotify error"); return(NULL);}
    int watch_d;
    memset(&event, 0 ,sizeof(struct inotify_event *));

    watch_d = inotify_add_watch(fd, path, IN_CREATE | IN_DELETE | IN_ATTRIB | IN_MOVED_TO | IN_MOVED_FROM | IN_MODIFY);
    if(watch_d < 0) {printf("%s ", path);perror("inotify error"); return(NULL);}
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
    pthread_t thread[2];
    t_log_file file_1, file_2;
    char *path[3] = {"/var/log/kern.log", "/var/log/auth.log", NULL};
    

    memset(thread, 0, sizeof(pthread_t) * 2);
    i = 0;
    while (i < 2)
    {
        if(pthread_create(&thread[i], NULL, monitor_file, path[i]) < 0)
        {
            perror("%s\n");
            perror("Error thread creation");

        }
        i++;
    }
    printf("I'am waiting for the trade %d\n", i);
    i = 0;
    while (i < 2)
    {
        pthread_join(thread[i],NULL);
        printf("Tread is back, nice %s\n", path[i]);

        i++;
    }
    return 0;
}

//0.027836354449391365
//0.01694159209728241
//0.014803417026996613