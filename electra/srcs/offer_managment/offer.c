#include "offer.h"

static char *create_offer_path(const char *str_1, const char *str_2 )
{
	char *path;
	int len_1, len_2;

	if(!str_1 || !str_2)
		return(NULL);
	len_2 = strlen(str_2);
	len_1 = strlen(str_1);
	path = malloc(sizeof(char) * (len_1 + len_2 + 1));
	if(!path){perror("Malloc"); return(NULL);}
	memcpy(path, str_1, len_1);
	memcpy(&path[len_1], str_2, len_2);
	path[len_2 + len_1] = '\0';
	return(path);
}

static void get_config_data(t_offre *node, xmlNodePtr root, int i)
{
	int len;
	
	if(!root)
		return;
	
	xmlNodePtr n = root;
	while (n)
	{
		if(n->type == XML_ELEMENT_NODE)
		{
			
			len = ft_strlen(n->name);
			//printf("%s ", n->name);
			// xmlNodeGetContent need to be free?
			if(!strncmp(n->name, "gpu_id", ft_strlen("gpu_id")))
			{
				if(i < GPU_MAX)
				{
					node->gpu[i] = atoi(xmlNodeGetContent(n));
					i+=1;
				}
			}
			else if(!strncmp(n->name, "id", len))
				node->id = atoi(xmlNodeGetContent(n));
			else if (!strncmp(n->name, "cpu", len))
				node->cpu= atoi(xmlNodeGetContent(n));
			else if (!strncmp(n->name, "vram", len))
				node->vram= atoi(xmlNodeGetContent(n));
			else if (!strncmp(n->name, "ram", len))
				node->ram= atol(xmlNodeGetContent(n));
			else if (!strncmp(n->name, "storage", len))
				node->storage= atoi(xmlNodeGetContent(n));

		}
		if(n->children)
			get_config_data(node, n->children, i);

		n = n->next;
	}
	return;
}


static t_offre *create_offer_node(struct dirent *dir_node)
{
	xmlDocPtr   doc;
	xmlNodePtr   root;
	t_offre     *node;
	char *offer_path;
	
	if(!dir_node)
		return(NULL);
	node = malloc(sizeof(t_offre));
	if(!node){perror("malloc create_offer_node");return(NULL);}
	offer_path = create_offer_path(OFFERS_DIR_PATH, dir_node->d_name);
	if(!offer_path){free(node); return(NULL);}

	if((doc = xmlReadFile(offer_path, NULL,0)) == NULL)
	{
		printf("Error read xml file\n");
		free(offer_path);
		free(node);
		return(NULL);
	}
	if((root = xmlDocGetRootElement(doc)) == NULL)
	{
		printf("Error get root node xml\n");
		free(offer_path);
		free(node);
		xmlFreeDoc(doc);
		return(NULL);
	}
	//printf("New XML START creation structure %s\n", dir_node->d_name);
	memset(node->gpu,0,sizeof(int) * GPU_MAX);
	get_config_data(node, root, 0);
	xmlFreeDoc(doc);
	
	node->xml_path = ft_strdup(offer_path);
	free(offer_path);

	node->name = ft_strdup(dir_node->d_name);
	node->resa_count = 0;
	node->next = NULL;
	return(node);
}

static void offer_add_back(t_offre **offers_list, t_offre *offer)
{
	
	t_offre *ptr;
	t_offre *precedent;

	if(!offers_list || !offer)
		return;
	if(!(*offers_list))
	{
		*offers_list = offer;
		return;
	}
	ptr = *offers_list;
	while (ptr->next)
	{
		precedent  = ptr;
		ptr = ptr->next;
	}
	if(ptr->id > offer->id)
	{
		precedent->next = offer;
		offer->next = ptr;
	}
	else
		ptr->next = offer;
}

void dislay_offer_list(t_offre **offre_list)
{
	t_offre *ptr;

	if(!offre_list)
		return;
	ptr = *offre_list;
	printf("%s\n", "────────────────── DISPLAY OFFER LIST ────────────────────");
	printf("%s\n", "──────────────────────────────────────────────────────────");
	while(ptr)
	{
		printf("│ %-12s │ %-40d │\n", "ID", ptr->id);
		printf("%s\n", "──────────────────────────────────────────────────────────");
		printf("│ %-12s │ %-40s │\n", "NAME", ptr->name);
		printf("│ %-12s │ %-40d │\n", "CPU", ptr->cpu);
		printf("│ %-12s │ %-40ld │\n", "RAM", ptr->ram);
		printf("│ %-12s │ %d %d %d %d %34s\n", "GPU", ptr->gpu[0], ptr->gpu[1], ptr->gpu[2], ptr->gpu[3], "|");
		printf("│ %-12s │ %-40d │\n", "VRAM", ptr->vram);
		printf("│ %-12s │ %-40ld │\n", "STORAGE", ptr->storage);
		printf("│ %-12s │ %-40d │\n", "RESA COUNT", ptr->resa_count);
		printf("│ %-12s │ %-40s │\n", "XML_PATH", ptr->xml_path);
		printf("%s\n", "──────────────────────────────────────────────────────────");

		ptr = ptr->next;
	}
	printf("\n");
}

int clean_list_offer(t_offre **offres)
{
	int     i;
	t_offre *tmp;

	if(offres)
	{
		i = 0;
		while (*offres)
		{

			tmp = (*offres);
			*offres = (*offres)->next;
			free(tmp->name);
			free(tmp->xml_path);
			free(tmp);
			i++;
		}
	}
	return (i);
}

t_offre **get_offer_list(void)
{
	FILE            *f;
	DIR             *dir;
	struct dirent   *dir_node;
	t_offre         **offers_list;
	t_offre         *offer;

	
	if((dir = opendir(OFFERS_DIR_OPEN))== NULL)
	{
		perror("open dir");
		return(NULL);
	}
	offers_list = malloc(sizeof(t_offre *));
	if(!offers_list){perror("malloc create_offer_list");closedir(dir);return(NULL);}
	*offers_list = NULL;
	while ((dir_node = readdir(dir)) != NULL)
	{
		if(strncmp(dir_node->d_name,".", strlen(dir_node->d_name)) == 0
		|| strncmp(dir_node->d_name,"..", strlen(dir_node->d_name)) == 0)
			continue;
		offer = create_offer_node(dir_node);
		if(!offer)
		{
			clean_list_offer(offers_list);
			return(NULL);
		}
		offer_add_back(offers_list, offer);
	}
	printf("\n");
	closedir(dir);
	return(offers_list);
}