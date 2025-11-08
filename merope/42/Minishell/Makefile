OS= $(shell uname)
ifeq ($(OS), Darwin)
	CC=cc
else
	CC=gcc
endif

GFLAGS= -Werror -Wextra -Wall -Wno-unused-parameter -g
BRANCH= $(shell git branch --show-current )
NAME=bin/minishell
NAME_TEST=bin/test
EXT_MOD=external_fonction
LIB= -Llibft -lft -lreadline
# Module readline : export EXT_F=readline

EXT_SRCS= 		srcs/main.c \
				srcs/envp/envp.c \
				srcs/tokeniser/tokeniser.c \
				srcs/ast/ast_generation.c \
				srcs/ast/ast_execution.c \
				srcs/ast/ast_tools.c \
				srcs/builtin/env.c \
				srcs/builtin/cd.c \
				srcs/builtin/pwd.c \
				srcs/builtin/echo.c \
				srcs/builtin/export.c \
				srcs/builtin/unset.c \
				srcs/builtin/exit.c \
				srcs/tokeniser/tokeniser_utils.c \
				srcs/tokeniser/tokeniser_args.c \
				srcs/tokeniser/tokeniser_redir.c \
				srcs/tokeniser/tokeniser_display.c \
				srcs/tokeniser/tokeniser_clean.c \
				srcs/tools/tools_path.c \

TEST_SRCS =  	test/main_test.c
				

 				
EXT_OBJ= $(EXT_SRCS:%.c=%.o)
TEST_OBJ= $(TEST_SRCS:%.c=%.o)
#########
#revoir cette parti

%.o:%.c
	@$(CC) $(GFLAGS) $< -c -o $@
#########

$(NAME): $(EXT_OBJ)
	@$(CC) $(EXT_OBJ)  $(LIB) -o $(NAME)

run: $(NAME)
ifeq ($(OS), Darwin)
	@./$(NAME)
else
	@echo $(CMD)
	./$(NAME)
#@ ./$(NAME)
endif

test: $(TEST_OBJ)
	@$(CC) $(TEST_OBJ) $(LIB) -o $(NAME_TEST)
	@./$(NAME_TEST)

# test: $(TEST_OBJ)
# 	@$(CC) $(TEST_OBJ) $(LIB) -o $(NAME_TEST)
# 	@valgrind --leak-check=full --log-file=valgrind/valgrind_test.log ./$(NAME_TEST)

clean:
	rm -rf $(EXT_OBJ)
	rm -rf $(TEST_OBJ)
#rm -f valgrind/valgrind.log
	rm -f valgrind/test/valgrind_test.log

fclean: clean
	rm -rf $(NAME)

re: fclean $(NAME)

git: fclean
	git add .
	git commit -m $(COM)
	git push origin $(BRANCH)

lib:
	cd libft && make fclean && make bonus && make clean
n:
	echo $(CC)

.phony:
	re lib git clean fclean