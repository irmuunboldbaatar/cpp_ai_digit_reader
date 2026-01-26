#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

class MNISTLoader {
   public:
    // Flips high-endian integers to little-endian
    uint32_t swap_endian(uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }

    // Reads MNIST images (idx3-ubyte format)
    std::vector<std::vector<double>> read_images(std::string path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open image file.");

        uint32_t magic_number, num_images, rows, cols;
        file.read((char*)&magic_number, 4);
        file.read((char*)&num_images, 4);
        file.read((char*)&rows, 4);
        file.read((char*)&cols, 4);

        magic_number = swap_endian(magic_number);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        std::vector<std::vector<double>> images(
            num_images, std::vector<double>(rows * cols));

        for (uint32_t i = 0; i < num_images; ++i) {
            for (uint32_t j = 0; j < rows * cols; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                // Normalize pixel values from [0, 255] to [0, 1] for the
                // neural network
                images[i][j] = static_cast<double>(pixel) / 255.0;
            }
        }
        return images;
    }

    // Reads MNIST labels (idx1-ubyte format)
    std::vector<int> read_labels(std::string path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open label file.");

        uint32_t magic_number, num_items;
        file.read((char*)&magic_number, 4);
        file.read((char*)&num_items, 4);

        magic_number = swap_endian(magic_number);
        num_items = swap_endian(num_items);

        std::vector<int> labels(num_items);
        for (uint32_t i = 0; i < num_items; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, 1);
            labels[i] = static_cast<int>(label);
        }
        return labels;
    }
};

#endif