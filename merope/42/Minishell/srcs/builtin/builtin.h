# ifndef BUILTIN_H
#define BUILTIN_H

#include "../../libft/libft.h"
#include "../envp/envp.h"
#include "../tokeniser/tokeniser.h"

#define EXPORT 0
#define UNSET 1

int ft_env(t_token *token, t_env *env);
int ft_cd(t_token *token);
int ft_pwd(void);
int ft_echo(t_token *token);
int ft_export(t_env *self_env, t_token *token);
int ft_unset(t_env *self_env, t_token *token);
void ft_exit();

int is_valide_variable(char *var);
size_t count_valide_variable(char **vars, int mode);
int is_on_env(char *var, char **env, int len);

#endif