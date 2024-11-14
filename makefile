all: clean
	gcc ./source/*.c -o main -lm

clean:
	rm -f main
