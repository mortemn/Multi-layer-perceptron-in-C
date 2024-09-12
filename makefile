all: clean
	gcc main.c math_utils.c -o main

clean:
	rm -f main
