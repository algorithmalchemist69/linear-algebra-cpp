CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wno-unused-variable
TARGET   = linalg

all: $(TARGET)

$(TARGET): main.cpp linalg.cpp linalg.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp linalg.cpp

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
