all:
	clang++ -std=c++23 -O3 -ffast-math src/main.cpp src/network.cpp -o main \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib \
	-lsfml-graphics -lsfml-window -lsfml-system
	./main