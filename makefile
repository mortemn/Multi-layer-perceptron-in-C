all: clean
	gcc ./source/*.c -o main

clean:
	rm -f main
