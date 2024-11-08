#define PROFILE  // turns on the reporting of timing results

#include "openfhe.h"
#include "ArgMapping.h"
#include <ctime>
#include <random>
#include <vector>
#include <algorithm> // For std::nth_element and std::partition

using namespace lbcrypto;
/*
input: the importance score
output: the pruning mask
other parameters to test this component:
  nsample: the number of data samples (and scores) 
  p: the keeping ratio (the sparsity of the mask)
*/
int partition(std::vector<double>& arr, int left, int right) {
    int pivot = arr[right];
    int i = left;
    for (int j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[right]);
    return i;
}

int quickselect(std::vector<double>& arr, int left, int right, int k) {
    if (left == right) {
        return arr[left];
    }
    int pivotIndex = partition(arr, left, right);
    if (k == pivotIndex) {
        return arr[k];
    } else if (k < pivotIndex) {
        return quickselect(arr, left, pivotIndex - 1, k);
    } else {
        return quickselect(arr, pivotIndex + 1, right, k);
    }
}

std::vector<int> genMask(std::vector<double>& arr, int k) {
    // Find the k-th largest element, note k is 0-based index
    int kthLargest = quickselect(arr, 0, arr.size() - 1, arr.size() - 1 - k);

    // Create a binary mask based on the k-th largest element
    std::vector<int> mask(arr.size(), 0);
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] >= kthLargest) {
            mask[i] = 1;
        }
    }
    return mask;
}

int main(int argc, char* argv[]) {
    /************* Argument Parsing  ************/
    /********************************************/
    uint32_t nsample = 43750;
    double p = 0.5;
    ArgMapping amap;
    amap.arg("nsample", nsample, "the number of data samples");
    amap.arg("p", p, "the keeping ratio");
    amap.parse(argc, argv);

    std::cout << "Dataset: number of samples          " << nsample << std::endl;
    std::cout << "Dataset: keeping ratio              " << p << std::endl;

    TimeVar t1;
    double timeAll(0.0);
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    std::vector<double> scores(nsample);
    for (size_t i = 0; i <nsample; i++) {
        scores[i] = dis(gen); // Generate and store the random number
    }
    int k = std::ceil(nsample*p);

    // compute client aided masking with the quick select in plaintext
    TIC(t1);
    std::vector<int> mask = genMask(scores, k - 1);
    timeAll = TOC(t1);

    size_t oneCnt = 0;
    for (size_t i = 0; i <nsample; i++) {
        if(mask[i] == 1) oneCnt++;
    }
    // std::cout << "[CAM] real keeping ratio: " << oneCnt*1.0 / nsample << std::endl;
    std::cout << "[CAM] evaluation time: " << timeAll << " ms" << std::endl;

    return 0;
}
