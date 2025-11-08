#include "dup.h"
#include <errno.h>
#include <assert.h>

#define PATH "valgrind/valgrind_test.log"

int err(int r,const char *fonction, char *module)
{
    printf("[ fonction : %s ][ module : %s ] Err :%s\n",fonction,module,strerror(r));
    return(r);
}


int try_dup()
{

    int new_fd;
    char read_buffer1[11];
    char read_buffer2[11];

    
    int r_fd = open(PATH, O_RDONLY);
    if(r_fd == -1)
        return(err(errno,__func__,"Open"));

    new_fd = dup(r_fd);
    if(new_fd == -1)
        return(err(errno,__func__,"Open"));

    if(read(r_fd, read_buffer1, 10) == -1)
        return(err(errno,__func__,"Open"));
    read_buffer1[11] = '\0';
    printf("buffer 1: %s\n",read_buffer1);

    // ls return the position of the cursor
    long ls = lseek(r_fd,-10, SEEK_CUR);
    printf("voici ls %ld\n",ls);

    if(read(new_fd, read_buffer2, 10) == -1)
        return(err(errno,__func__,"Open"));
    read_buffer2[11] = '\0';
    printf("buffer 2 : %s\n",read_buffer2);
    assert(!ft_strncmp(read_buffer1,read_buffer2, 10));
    return(0);
}

