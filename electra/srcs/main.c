#include "vm_managment/vm.h"
#include "offer_managment/offer.h"




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




// iso 8601     AAAA-MM-JJTHH:MM:SS,ss-/+FF:ff

int electra_vm_ochestrator()
{
    t_resa **resa_tree;
    t_resa *resa,*resa2;

    char *args[] = {"2029-07-14T20:12:12", "1", "2"};
    char *args2[] = {"2027-07-14T07:12:12", "2", "3"};
    resa_tree = malloc(sizeof(t_resa *));
    if(!resa_tree){perror("malloc"); return(1);}

    // transforme iso date to resa_stuct

    resa = iso_date_to_resa(args[0], args[1], args[2]);
    resa2 = iso_date_to_resa(args2[0], args2[1], args2[2]);
    //resa3 = iso_date_to_resa(args3[0], args3[1], args3[2]);
    // if(!resa)
    // {
    //     printf("Error resa\n");
    //     free(resa_tree);
    //     return(1);
    // }
    display_resa(resa);
    display_resa(resa2);
    //display_resa(resa3);
    free(resa);
    free(resa2);
    free(resa_tree);
    return(0);

}

int main(void)
{

    electra_vm_ochestrator();
    
    
    return(0);
}