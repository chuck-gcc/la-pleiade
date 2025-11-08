#include "ast.h"

int open_redirection(t_token *ast)
{
    char *redir;

    if(!ast->redir[0])
        return(0);
    redir = ast->redir[0];
    if(ft_strncmp(redir,"<", ft_strlen_longest("<", redir)) == 0)
        return(1);
    return(0);
}



// pid_t execute_heredoc(t_token *ast, int tub[2], char **envp)
// {

//     pid_t pid;
//     char *delim;
//     int *tube;

//     tube = tub;
//     if(pipe(tube) == -1) {perror("pipe"); return(1);}
//     pid = fork();
//     if(pid == -1){perror("fork"); return(1);}
//     else if(pid == 0)
//     {
//         int b_read;
//         delim = ast->radir[1];
//         close(tube[0]);
//         do
//         {
//             char *line = readline("> ");
//             if(ft_nbr_of_word(line) == 1)
//             {
//                 if(ft_strncmp(line, delim, ft_strlen_longest(line, delim)) == 0)
//                     exit(0);
//             }
//             b_read = write(tube[1], line, ft_strlen(line));
//             if(b_read == -1)
//             {
//                 perror("read");
//                 exit(1);
//             }
//             write(tube[1], "\n", ft_strlen("\n"));
//         } while (b_read > 0);
//         exit(errno);
//     }
//     return(pid);
// }