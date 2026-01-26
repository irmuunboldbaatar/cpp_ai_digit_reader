#include <fstream>
#include <iostream>
#include <vector>

#include "mnist_loader.h"
#include "network.h"
using namespace std;

int main() {
    MNISTLoader loader;
    vector<vector<double>> train_images;
    vector<int> train_labels;
    vector<vector<double>> check_images;
    vector<int> check_labels;
    try {
        train_images = loader.read_images("data/train-images.idx3-ubyte");
        train_labels = loader.read_labels("data/train-labels.idx1-ubyte");
        check_images = loader.read_images("data/t10k-images.idx3-ubyte");
        check_labels = loader.read_labels("data/t10k-labels.idx1-ubyte");
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    // Want to add feature that loads sizes and names from network_config.txt
    Network net({0});
    string model_name = "784_100_10_L2";
    net.load("model/model_data_" + model_name + ".txt");

    int epochs;
    int min_batch_size;
    double eta;
    // Loading from training_config.txt
    ifstream training_config("training_config.txt");
    if (!training_config) {
        cerr << "Error opening training config file!\
        Using default values instead."
        << endl;
        // Default config values.
        int epochs = 10;
        int min_batch_size = 100;
        double eta = 1.0;
    } else {
        training_config >> epochs;
        training_config >> min_batch_size;
        training_config >> eta;
    }
    training_config.close();
    // Training start!
    cout << "model: " << model_name << endl;
    cout << "epochs: " << epochs << " mini_batch_size: " << min_batch_size
         << " learning rate: " << eta << endl;
    cout << "Training started!" << endl;
    net.SGD(epochs, min_batch_size, eta, train_images, train_labels,
            check_images, check_labels);
    net.save("model/model_data_" + model_name + ".txt");
    return 0;
}

/*
Usage:


    vector<int> layer_sizes = {784, 30, 10};
    Network net(layer_sizes);
    net.save("model_data.txt");


    Network loadedNet({0});
    loadedNet.load("model_data.txt");


*/