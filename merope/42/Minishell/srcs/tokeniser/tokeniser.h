# ifndef TOKENISER_H
# define TOKENISER_H

#include<stdio.h>
#include <dirent.h>
#include <assert.h>
#include "../../libft/libft.h"
#include "../tools/tools.h"



/* 
    TOKEN:

        *command
        *arg
        *builtin
        *pipe
        *redirection left
        *redirection right
        *single quote
        *double cote
        *word
*/

#define CMD             1
#define WORD            2
#define BUILTIN         3
#define PIPE            4 //'|'
#define REDIR_LEFT      5 //'<'    
#define REDIR_RIGHT     6 //'>'
#define REDIR_APPEND    7 //'>>'
#define DELIM           8 //'<<'
#define VAR             9


typedef struct s_token
{
    int             num;
    int             type;
    int             precedence;
    int             asso;
    char            *value;
    char            **args;
    char            *redir[2];
    int             redir_type;
    struct s_token  *left;
    struct s_token  *right;

} t_token;



t_list  **get_token_list(char *str, t_list **lst, char **envp, int status_code);

// utils

int     ft_is_builtin(char *str);
int     ft_is_commande(char *str);
int     get_token_type(char *str);
int     get_asso(int token_type);
int     get_precedence(int token_type);

// tokeniser redir
int get_redir(t_list *node, char **input);
int is_redir(char *str);

// tokeniser args
int get_args(t_list *node, char **input, char **envp, int status_code);
int ft_expend_var(t_token *token, char **envp, int status_code);
//tokeniser display

char    *print_token_type(int token_type);
void    display_content_lst(void *liste);
void    display_args_of_cmd(void *liste);
void    display_arg_of_cmd(t_token *token);

// cleannig
void delete_list(void *content);

# endif