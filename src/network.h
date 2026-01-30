#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
using namespace std;

class Network {
   public:
    Network(vector<int> v);
    void SGD(int epochs, int batch_size, double eta,
             vector<vector<double>>& train_images, vector<int>& train_labels,
             vector<vector<double>>& check_images, vector<int>& check_labels);
    void save(string filename);
    void load(string filename);
    void evaluate(vector<double>& img);
    vector<double> get_eval();
    int get_value();
    void update_mini_batch(const vector<int>& indices, int start,
                           int mini_batch_size, double eta,
                           const vector<vector<double>>& train_images,
                           const vector<int>& train_labels);

   private:
    vector<int> sizes;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    int value;
    vector<double> eval;
};

#endif