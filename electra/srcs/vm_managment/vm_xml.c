#include "vm.h"

char *get_xml(const char *path)
{
    FILE *f;
    char *xml;
    long len;

    if((f = fopen(path, "r")) < 0)
    {
        perror("fopen");
        return(NULL);
    }
    fseek(f,0, SEEK_END);
    len = ftell(f);
    if(len <= 0)
        return(NULL);
    xml = malloc(sizeof(char) * (len + 1));
    if(!xml)
    {
        perror("malloc");
        fclose(f);
        return(NULL);
    }
    rewind(f);
    if(fread(xml,sizeof(char),len,f) <= 0)
    {
        perror("fread");
        return(NULL);
    }
    xml[len] = '\0';
    fclose(f);
    return(xml);
}
