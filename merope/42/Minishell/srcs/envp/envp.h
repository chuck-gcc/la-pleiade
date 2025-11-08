#ifndef ENVP_H
#define ENVP_H

#include "../../libft/libft.h"
#include <assert.h>


typedef struct s_env_manager
{
    char ***env;
    int   start;

    int     (*swap_env)(struct s_env_manager *self ,char **new_env);
    int     (*dup_env)(struct s_env_manager *self ,char **new_env);
    int     (*destroy_env)(struct s_env_manager *self);
    void    (*print_env)(struct s_env_manager *self);
    void    (*display_export_env)(struct s_env_manager *self);

} t_env;

t_env *init_env(char **envp);

#endif