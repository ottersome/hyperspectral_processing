all: main 

main:
	python main.py

clean:
	rm --force data/cache/*
	rm --force models/*
