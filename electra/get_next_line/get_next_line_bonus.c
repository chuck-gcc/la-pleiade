/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   get_next_line_bonus.c                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cbouhadr <cbouhadr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/11/10 12:45:21 by cbouhadr          #+#    #+#             */
/*   Updated: 2024/11/10 13:32:23 by cbouhadr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "get_next_line_bonus.h"

char	*read_traitement(int fd, char *buff, char *line, char **stach);

char	*clean(char **old_ptr, char *new)
{
	free (*old_ptr);
	*old_ptr = new;
	return (*old_ptr);
}

char	*ft_process(int fd, char *buff, char *line, char **stach)
{
	line = clean(&line, gl_strjoin(line, buff));
	if (!line)
		return (NULL);
	if (gl_len_index_of(buff, 'l') > 0)
		line = read_traitement(fd, buff, line, stach);
	return (line);
}

char	*read_traitement(int fd, char *buff, char *line, char **stach)
{
	char	*t;
	char	*tmp;
	int		b;
	int		index_return;

	b = read(fd, buff, BUFFER_SIZE);
	if (b < 0)
	{
		free(line);
		return (NULL);
	}
	buff[b] = '\0';
	if (b == 0)
		return (line);
	index_return = gl_len_index_of(buff, 'i');
	if (index_return == -1)
		return (ft_process(fd, buff, line, stach));
	tmp = gl_substr(buff, 0, index_return + 1);
	line = clean(&line, gl_strjoin(line, tmp));
	free(tmp);
	t = gl_strchr(buff, 10) + 1;
	*stach = gl_strdup(t);
	return (line);
}

char	*ft_stach_processing(char **stach, char **line)
{
	char	*tmp1;
	int		idx_return;
	int		len_return;

	idx_return = gl_len_index_of(*stach, 'i') + 1;
	len_return = gl_len_index_of(*stach, 'l');
	tmp1 = gl_substr(*stach, 0, idx_return);
	*line = gl_strjoin(*line, tmp1);
	if (!(*line))
	{
		free(tmp1);
		return (NULL);
	}
	tmp1 = clean(&tmp1, gl_substr(*stach, idx_return, len_return - idx_return));
	free(*stach);
	if (gl_len_index_of(tmp1, 'l') == 0)
	{
		free(tmp1);
		*stach = NULL;
	}
	else
		*stach = tmp1;
	return (*line);
}

char	*get_next_line(int fd)
{
	char		*buffer;
	char		*line;
	static char	*stach[200];

	if (fd < 0 || BUFFER_SIZE <= 0)
		return (NULL);
	buffer = malloc(sizeof(char) * (BUFFER_SIZE + 1));
	if (!buffer)
		return (NULL);
	line = NULL;
	if (stach[fd] && gl_len_index_of(stach[fd], 'i') > -1)
	{
		free(buffer);
		return (ft_stach_processing(&stach[fd], &line));
	}
	else if (stach[fd] && gl_len_index_of(stach[fd], 'i') == -1)
	{
		line = clean(&line, gl_strjoin(line, stach[fd]));
		free(stach[fd]);
		stach[fd] = NULL;
	}
	line = read_traitement(fd, buffer, line, &stach[fd]);
	free(buffer);
	return (line);
}

// int main(void)
// {
// 	int fd1;
// 	int fd2;

// 	fd1 = open("file1.txt",O_RDONLY);
// 	if(fd1 == -1)
// 	{
// 		printf("erreur\n");
// 		return (-1);
// 	}
// 	fd2 = open("file2.txt",O_RDONLY);
// 	if(fd2 == -1)
// 	{
// 		printf("erreur\n");
// 		return (-1);
// 	}
// 	printf("%s\n", get_next_line(fd1));
// 	printf("%s\n", get_next_line(fd2));
// 	printf("%s\n", get_next_line(fd1));
// 	printf("%s\n", get_next_line(fd2));
// 	printf("%s\n", get_next_line(fd1));
// 	return(0);
// }
