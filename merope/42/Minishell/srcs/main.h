#ifndef MAIN_H
#define MAIN_H

#include "tokeniser/tokeniser.h"
#include "builtin/builtin.h"
#include "tools/tools.h"
#include "ast/ast.h"
#include "envp/envp.h"
#include "../libft/libft.h"
#include "../test/external_fonction/dup.h"
#include "../test/external_fonction/readline.h"

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
#include <unistd.h>
#include <readline/readline.h>
#include <readline/history.h>
 #include <termios.h>
#include <signal.h>


#define START printf("Start\n")
#define END printf("End\n")
 
#endif 