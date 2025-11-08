#ifndef TOOL_H
#define TOOL_H

#include "../../libft/libft.h"

#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/time.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>

char    *get_os(void);
char    *get_base_path(char *str);
char    *get_path(char *str);


#endif