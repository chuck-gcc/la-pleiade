#ifndef AST_H
#define AST_H


#include "../builtin/builtin.h"
#include "../envp/envp.h"
#include "../tools/tools.h"
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
#include <readline/readline.h>
#include <dirent.h>


int     generate_ast(t_list *token_list, t_token **ast_root);
int     execute_ast(t_token *ast, t_env *env, int saved);
void    display_binary_tree(t_token *parent, t_token *actual, int deriv);
char    *get_os(void);
char    *get_base_path(char *str);

int     open_redirection(t_token *ast);
pid_t   execute_heredoc(t_token *ast, int tub[2], char **envp);

#endif