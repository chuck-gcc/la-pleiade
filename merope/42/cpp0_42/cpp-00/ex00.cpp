/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ex00.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: cw3l <cw3l@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/19 01:47:10 by cw3l              #+#    #+#             */
/*   Updated: 2025/06/20 10:39:42 by cw3l             ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <iostream>
#include <cstring>

int main(int argc, char **argv) {
    
    int i;
    int j;
    char *str;

    i = 1;
    j = 0;
    if(argc < 1)
        return(1);
    else if(argc == 1)
        std::cout << "* LOUD AND UNBEARABLE FEEDBACK NOISE *";
    while(i < argc)
    {
        str = argv[i];
        for(j = 0; j < std::strlen(argv[i]); j++)
            std::cout << (char)toupper(argv[i][j]) ;
        std::cout << " ";
        i++;
    }
    std::cout << "\n";

    return 0;
}