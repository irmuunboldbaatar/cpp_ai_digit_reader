#include <iostream>
#include <string>

#include "mnist_loader.h"
#include "network.h"
using namespace std;

int main() {
    string model_name = "784_100_10";
    Network net({0});
    net.load("model/model_data_" + model_name + ".txt");
    MNISTLoader loader;
    vector<vector<double>> check_images;
    vector<int> check_labels;
    try {
        check_images = loader.read_images("data/t10k-images.idx3-ubyte");
        check_labels = loader.read_labels("data/t10k-labels.idx1-ubyte");
        int total = check_images.size();
        int correct = 0;
        cout << "Testing model: " << model_name << endl;
        cout << "Loaded " << total << " test images." << endl;
        cout << "Test start!" << endl;
        for (int i = 0; i < total; i++) {
            net.evaluate(check_images[i]);
            if (net.getValue() == check_labels[i]) {
                correct++;
            }
            cout << "\rProgress: [" << string(50 * (i + 1) / total, '#')
                 << string(50 - 50 * (i + 1) / total, ' ') << "]";
            cout << " " << ((i + 1) * 100) / total << "%";
            cout << flush;
        }
        cout << endl << "Result: " << correct << "/" << total << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}