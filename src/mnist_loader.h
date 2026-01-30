#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

class MNISTLoader {
   public:
    // Flips high-endian integers to little-endian
    uint32_t swap_endian(uint32_t val);
    // Reads MNIST images (idx3-ubyte format)
    std::vector<std::vector<double>> read_images(std::string path);
    // Reads MNIST labels (idx1-ubyte format)
    std::vector<int> read_labels(std::string path);
};

#endif