main: main.c
	gcc -std=c99 -Wall -O3 $^ -o $@ -lm
