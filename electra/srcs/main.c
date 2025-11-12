#include "vm_managment/vm.h"
#include "offer_managment/offer.h"

struct tm date_obj(t_date date)
{

    struct tm t;

    t.tm_mday = (int)date.day;
    t.tm_mon = (int)date.mount;
    t.tm_year = (int)date.year;
    t.tm_hour = (int)date.hour;
    t.tm_min = (int)date.min;
    t.tm_sec = (int)date.sec;
    return(t);
}


char **read_line()
{
    int fd;
    char **split;
    char buffer[1024];

    printf("add vm reservation ");
    fflush(stdout);

    fd = read(STDIN_FILENO, buffer, 1023);
    if(fd == -1){perror("read");return(NULL);}
    buffer[fd] = '\0';

    split = ft_split(buffer, 32);
    if(!split)
    {
        return(NULL);
    }
    return(split);
}


t_resa *iso_date_to_resa(const char *iso_date, char *id, char *time)
{
    char **split_time;
    char **split_date;
    char **split_hour;
    char *date;
    char *hour;
    t_resa *resa;
    struct tm t;

    if(!iso_date)
        return(NULL);
    split_time = ft_split(iso_date, 'T');
    if(!split_time)
        return(NULL);
    date = split_time[0];    
    hour = split_time[1];
    split_date = ft_split(date,'-');
    if(!split_date){ft_split_clean(&split_time); return(NULL);}
    split_hour = ft_split(hour,':');
    if(!split_hour){ft_split_clean(&split_date);ft_split_clean(&split_time); return(NULL);}


    resa = malloc(sizeof(t_resa));
    if(!resa){perror("malloc resa"); ft_split_clean(&split_hour);ft_split_clean(&split_date);ft_split_clean(&split_time); return(NULL);}

    //printf("voici %d-%d-%dT%d:%d:%d\n",atoi(split_date[0]),atoi(split_date[1]), atoi(split_date[2]),atoi(split_hour[0]),atoi(split_hour[1]),atoi(split_hour[2]));
    
    t.tm_year = atoi(split_date[0]) - 1900;
    t.tm_mon = atoi(split_date[1]);
    t.tm_mday = atoi(split_date[2]);

    t.tm_hour = atoi(split_hour[0]);
    t.tm_min = atoi(split_hour[1]);
    t.tm_sec = atoi(split_hour[2]);
    
    
    resa->start = mktime(&t);   
    resa->offer_id = ft_atoi(id);
    resa->end =  resa->start + (3600 *  atoi(time));
    
    
    ft_split_clean(&split_hour);
    ft_split_clean(&split_date);
    ft_split_clean(&split_time);
    return(resa);    
}

// iso 8601     AAAA-MM-JJTHH:MM:SS,ss-/+FF:ff

int electra_vm_ochestrator()
{
    t_resa **resa_tree;
    t_resa *resa;

    char *args[] = {"2025-10-29T20:12:12", "1", "2"};
    resa_tree = malloc(sizeof(t_resa *));
    if(!resa_tree){perror("malloc"); return(1);}

    // transforme iso date to resa_stuct

    resa = iso_date_to_resa(args[0], args[1], args[2]);
    if(!resa)
    {
        printf("Error resa\n");
        free(resa_tree);
        return(1);
    }
    else
    {
        
        display_resa(resa);
    }
    free(resa_tree);
    return(0);

}

int main(void)
{

    electra_vm_ochestrator();
    
    // t_offre **list;
    // list =  get_offer_list();

    
    // time_t now;
    // now = time(NULL);
    // struct tm *t = localtime(&now);
    

    

    // *resa_tree = NULL;
    // create_resa(t, 2, 0,resa_tree);
    
    // display_resa(*resa_tree);
    // free(list);
    // free(resa_tree);
    return(0);
}