#include <algorithm>  // for std::shuffle
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "network.h"

using namespace std;

// member variables:                            // use case:
// vector<int> sizes;                           // sizes[layer]
// vector<vector<vector<double>>> weights;      // weights[layer][this layer's
//                                              index][previous layer's index]
// vector<vector<double>> biases;               // biases[layer][this layer's
//                                              index]
// int value; vector<double> eval;

double squish(double x) { return 1.0 / (1.0 + exp(-x)); }

double squish_prime(double x) {
    double s = squish(x);
    return s * (1.0 - s);
}

vector<double> Network::get_eval() { return eval; }

int Network::get_value() { return value; }

Network::Network(vector<int> v) : sizes(v) {
    random_device rd;
    mt19937 gen(rd());

    for (int layer = 1; layer < sizes.size(); layer++) {
        double limit = sqrt(1.0 / sizes[layer - 1]);
        uniform_real_distribution<> dis(-limit, limit);
        vector<double> layer_bias(sizes[layer]),
            neuron_weight(sizes[layer - 1]);
        vector<vector<double>> layer_weight(sizes[layer]);
        for (int i = 0; i < sizes[layer]; i++) {
            layer_bias[i] = dis(gen);
            for (int j = 0; j < sizes[layer - 1]; j++) {
                neuron_weight[j] = dis(gen);
            }
            layer_weight[i] = neuron_weight;
        }
        weights.push_back(layer_weight);
        biases.push_back(layer_bias);
    }

    eval = {};
    for (int i = 0; i < 10; i++) {
        eval.push_back(0.0);
    }
}

// vector dot product
double dot(vector<double>& a, vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument(
            "Error: Cannot calculate dot product.\n\
                vector<double> a, vector<double> b sizes doesn't match!");
    }
    double sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// matrix and vector multiplication
vector<double> mdot(vector<vector<double>>& A, vector<double>& b) {
    if (A[0].size() != b.size()) {
        throw invalid_argument(
            "Cannot calculate matrix and vector product.\n\
                vector<vector<double>> A, vector<double> b sizes doesn't match!");
    }
    vector<double> product(A.size());
    for (int i = 0; i < product.size(); i++) {
        product[i] = dot(A[i], b);
    }
    return product;
}

// vector addition
vector<double> sum(vector<double>& a, vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument(
            "Error: Cannot calculate vector addition.\n\
                vector<double> a, vector<double> b sizes doesn't match!");
    }
    vector<double> sum(a.size());
    for (int i = 0; i < a.size(); i++) {
        sum[i] = a[i] + b[i];
    }
    return sum;
}

void Network::evaluate(vector<double>& input) {
    // feed-forward
    vector<double> curr = input;
    for (int layer = 1; layer < sizes.size(); layer++) {
        vector<double> next(sizes[layer]);
        for (int n = 0; n < sizes[layer]; n++) {
            next[n] = squish(dot(curr, weights[layer - 1][n]) +
                             biases[layer - 1][n]);
        }
        curr = next;
    }

    // log output
    eval = {};
    for (int i = 0; i < curr.size(); i++) {
        eval.push_back(curr[i]);
    }

    // find maximum
    double max_weight = -1.0;
    int predicted_digit = 0;
    for (int i = 0; i < curr.size(); i++) {
        if (curr[i] > max_weight) {
            max_weight = curr[i];
            predicted_digit = i;
        }
    }
    value = predicted_digit;
    return;
}

void Network::save(string filename) {
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Error opening file for writing!" << endl;
        return;
    }

    // 1. Save sizes
    outFile << sizes.size() << "\n";
    for (int s : sizes) outFile << s << " ";
    outFile << "\n";

    // 2. Save weights (3D: vector of matrices)
    outFile << weights.size() << "\n";
    for (const auto& matrix : weights) {
        outFile << matrix.size() << " "
                << (matrix.empty() ? 0 : matrix[0].size()) << "\n";
        for (const auto& row : matrix) {
            for (double val : row) outFile << val << " ";
            outFile << "\n";
        }
    }

    // 3. Save biases (2D: vector of vectors)
    outFile << biases.size() << "\n";
    for (const auto& vec : biases) {
        outFile << vec.size() << "\n";
        for (double val : vec) outFile << val << " ";
        outFile << "\n";
    }

    outFile.close();
}

void Network::load(string filename) {
    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Error opening file for reading!" << endl;
        return;
    }

    // 1. Load sizes
    size_t numLayers;
    inFile >> numLayers;
    sizes.resize(numLayers);
    for (size_t i = 0; i < numLayers; ++i) inFile >> sizes[i];

    // 2. Load weights
    size_t numWeightMatrices;
    inFile >> numWeightMatrices;
    weights.resize(numWeightMatrices);
    for (size_t i = 0; i < numWeightMatrices; ++i) {
        size_t rows, cols;
        inFile >> rows >> cols;
        weights[i].resize(rows, vector<double>(cols));
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                inFile >> weights[i][r][c];
            }
        }
    }

    // 3. Load biases
    size_t numBiasVectors;
    inFile >> numBiasVectors;
    biases.resize(numBiasVectors);
    for (size_t i = 0; i < numBiasVectors; ++i) {
        size_t bSize;
        inFile >> bSize;
        biases[i].resize(bSize);
        for (size_t j = 0; j < bSize; ++j) {
            inFile >> biases[i][j];
        }
    }

    inFile.close();
}

void Network::SGD(int epochs, int mini_batch_size, double eta,
                  vector<vector<double>>& train_images,
                  vector<int>& train_labels,
                  vector<vector<double>>& check_images,
                  vector<int>& check_labels) {
    vector<int> indices(train_images.size());
    for (int i = 0; i < indices.size(); ++i) indices[i] = i;

    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), g);
        for (int i = 0; i < train_images.size() - mini_batch_size;
             i += mini_batch_size) {
            
            update_mini_batch(indices, i, mini_batch_size, eta, train_images,
                              train_labels);
            
        }
        cout << "epoch " << epoch << " completed. ";

        int total = check_images.size();
        int correct = 0;
        for (int i = 0; i < total; i++) {
            evaluate(check_images[i]);
            if (value == check_labels[i]) {
                correct++;
            }
        }
        cout << "accuracy: " << correct << "/" << total << endl;
    }
}

vector<double> distort_image(const vector<double>& img) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);
    uniform_real_distribution<> scl(0.9, 1.1);
    uniform_real_distribution<> rot(-0.1, 0.1);
    double tx = dis(gen);
    double ty = dis(gen);
    double scale = scl(gen);
    double angle = rot(gen);
    
    const int size = 28;
    vector<double> out(size * size, 0.0); // Initialize with black background (0.0)

    double cos_a = cos(angle);
    double sin_a = sin(angle);

    for (int y_out = 0; y_out < size; ++y_out) {
        for (int x_out = 0; x_out < size; ++x_out) {
            
            // 1. Center the coordinates (shift origin to 14,14)
            double x_centered = x_out - 13.5;
            double y_centered = y_out - 13.5;

            // 2. Apply inverse transformation (Map output back to input)
            // Scaling and Rotation applied relative to center
            double x_in = (cos_a * x_centered + sin_a * y_centered) / scale + 13.5 - tx;
            double y_in = (-sin_a * x_centered + cos_a * y_centered) / scale + 13.5 - ty;

            // 3. Bilinear Interpolation
            int x0 = floor(x_in);
            int y0 = floor(y_in);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 >= 0 && x1 < size && y0 >= 0 && y1 < size) {
                double dx = x_in - x0;
                double dy = y_in - y0;

                double val = img[y0 * size + x0] * (1 - dx) * (1 - dy) +
                             img[y0 * size + x1] * dx * (1 - dy) +
                             img[y1 * size + x0] * (1 - dx) * dy +
                             img[y1 * size + x1] * dx * dy;

                out[y_out * size + x_out] = val;
            }
        }
    }
    return out;
}

void Network::update_mini_batch(const vector<int>& indices, int start,
                                int mini_batch_size, double eta,
                                const vector<vector<double>>& train_images,
                                const vector<int>& train_labels) {
    vector<vector<vector<double>>> nabla_w; 
    vector<vector<double>> nabla_b;
    for (int layer = 1; layer < sizes.size(); layer++) {
        nabla_b.push_back(vector<double>(sizes[layer], 0.0));
        nabla_w.push_back(vector<vector<double>>(sizes[layer], vector<double>(sizes[layer-1], 0.0)));
    }
    
    int num_layers = sizes.size();
    for (int i = start; i < start + mini_batch_size; i++) {
        int img_index = indices[i];

        // feed-forward
        vector<vector<double>> Z(num_layers - 1);
        vector<vector<double>> A(num_layers);
        A[0] = train_images[img_index];
        vector<double> activation = A[0];

        for (int layer = 1; layer < sizes.size(); layer++) {
            Z[layer - 1].resize(sizes[layer]);
            A[layer].resize(sizes[layer]);

            for (int j = 0; j < sizes[layer]; j++) {
                Z[layer - 1][j] = dot(activation, weights[layer - 1][j]) +
                                  biases[layer - 1][j];
                A[layer][j] = squish(Z[layer - 1][j]);
            }
            activation = A[layer];
        }

        // output error
        int L = num_layers - 1;
        vector<double> delta(sizes[L]);
        for (int j = 0; j < sizes[L]; j++) {
            // calculating errors
            double expectation = (train_labels[img_index] == j ? 1.0 : 0.0);
            delta[j] = A[L][j] - expectation;

            // accumulating gradients
            nabla_b[L-1][j] += delta[j];
            for (int k = 0; k < sizes[L-1]; k++) {
                nabla_w[L-1][j][k] += delta[j] * A[L-1][k];
            }
        }

        // back propagation
        for (int layer = L - 1; layer > 0; layer--) {
            vector<double> next_delta(sizes[layer]);
            for (int j = 0; j < sizes[layer]; j++) {
                double error = 0;
                for (int k = 0; k < sizes[layer+1]; k++) {
                    error += weights[layer][k][j] * delta[k];
                }
                next_delta[j] = error * squish_prime(Z[layer-1][j]);
            }
            delta = next_delta;

            // accumulating gradients for hidden layers
            for (int j = 0; j < sizes[layer]; j++) {
                nabla_b[layer-1][j] += delta[j];
                for (int k = 0; k < sizes[layer-1]; k++) {
                    nabla_w[layer-1][j][k] += delta[j] * A[layer-1][k];
                }
            }
        }
    }

    double lambda = 5.0; // L2 regularization parameter
    int n = train_images.size();
    for (int l = 0; l < weights.size(); l++) {
        for (int j = 0; j < weights[l].size(); j++) {
            biases[l][j] -= (eta / mini_batch_size) * nabla_b[l][j];
            for (int k = 0; k < weights[l][j].size(); k++) {
                double weight_decay = 1.0 - (eta * (lambda / n));
                weights[l][j][k] = (weight_decay * weights[l][j][k]) - 
                                   ((eta / mini_batch_size) * nabla_w[l][j][k]);
            }
        }
    }
}
