#include "../include/electra/offer.h"

#define HourToMs(hour) (hour * (60 * (60 * 1000)))




int add_resa(t_resa *root, t_resa *resa)
{
    t_resa *tmp;

    if(!root || !resa)
        return(1);
    if(!root->left)
    {
        root->left = resa;
        return(0);
    }
    if(!root->right)
    {
        root->right = resa;
        return(0);
    }
    else
    {
        tmp = root;
        root = resa;
        resa->left = tmp;
        return(0);
    }
    return(1);
}


void printf_date(time_t *start, time_t *end)
{
    struct tm date_s;
    struct tm date_e;

    date_s = *localtime(start);
    date_e = *localtime(end);
    
    printf("START ---> %d/%d/%d %d:%d:%d\n", date_s.tm_mday, date_s.tm_mon, 1900 + date_s.tm_year, date_s.tm_hour, date_s.tm_min,date_s.tm_sec);
    printf("END -----> %d/%d/%d %d:%d:%d\n", date_e.tm_mday, date_e.tm_mon, 1900 + date_e.tm_year, date_e.tm_hour, date_e.tm_min,date_e.tm_sec);
    printf("\n\n");
}

void display_resa(t_resa *root)
{
    if(!root)
        return;
    display_resa(root->left);
    printf("ID: %d\n", root->offer_id);
    printf_date(&(root->start), &(root->end));
    display_resa(root->right);
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
    ft_bzero(resa,sizeof(t_resa));
    ft_bzero(&t,sizeof(struct tm));
    
    t.tm_year = atoi(split_date[0]) - 1900;
    t.tm_mon = atoi(split_date[1]);
    t.tm_mday = atoi(split_date[2]);

    t.tm_hour = atoi(split_hour[0]);
    t.tm_min = atoi(split_hour[1]);
    t.tm_sec = atoi(split_hour[2]);
    
    
    resa->start = 0;   
    resa->start = mktime(&t);   
    resa->offer_id = ft_atoi(id);
    resa->end =  resa->start + (3600 *  atoi(time));
    
    
    ft_split_clean(&split_hour);
    ft_split_clean(&split_date);
    ft_split_clean(&split_time);
    return(resa);    
}