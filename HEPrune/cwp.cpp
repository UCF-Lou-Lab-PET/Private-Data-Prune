#define PROFILE  // turns on the reporting of timing results

#include "openfhe.h"
#include "ArgMapping.h"
#include <ctime>
#include <random>
#include <deque>

using namespace lbcrypto;
/*
input: ciphertext queue, prunining mask
output: the importance score.
other parameters to test this component:
  batch: batch_size
  ctn: number of ciphertexts
  p: the keeping ratio (the sparsity of the mask)
*/

int main(int argc, char* argv[]) {

    /************* Argument Parsing  ************/
    /********************************************/
    uint32_t batch = 128;
    uint32_t ctn = 96;
    double p = 0.5;
    ArgMapping amap;
    amap.arg("batch", batch, "batchsize in packing");
    amap.arg("ctn", ctn, "the number of ciphertexts");
    amap.arg("p", p, "the keeping ratio");
    amap.parse(argc, argv);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(12);
    parameters.SetScalingModSize(50);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    uint32_t batchSize = 32768;

    auto ringDim = cc->GetRingDimension();
    std::cout << "CKKS scheme is using ring dimension " << ringDim<< std::endl << std::endl;
    std::cout << "sec:" << parameters.GetSecurityLevel() <<std::endl;
    uint32_t maxNumSlots = ringDim / 2;
    std::cout << "CKKS scheme number of slots         " << batchSize << std::endl;
    std::cout << "CKKS scheme max number of slots     " << maxNumSlots << std::endl;
    std::cout << "Dataset: number of ciphertexts      " << ctn << std::endl;
    std::cout << "Dataset: keeping ratio              " << p << std::endl;

    uint32_t msg_cnt = ctn;
    uint32_t msg_length = batchSize;
    TimeVar t1;
    double timeAll(0.0);
    // TimeVar t2;
    // double timeEval(0.0);
    uint32_t rot_cnt=0;
    std::random_device rd;  // Random device
    std::mt19937 gen(rd()); // Mersenne Twister generator
    std::uniform_real_distribution<> dis1(-1.0, 1.0);

    std::vector<std::vector<double>> msg_d(msg_cnt, std::vector<double>(msg_length));
    for (uint32_t i = 0; i < msg_cnt; ++i) {
        for (size_t j = 0; j < msg_length; ++j) {
            msg_d[i][j] = dis1(gen);
        }
    }
    std::vector<int> rot_loc;
    // rot_loc.push_back(1);
    for (uint32_t i = 1; i < batch; ++i) {
      rot_loc.push_back(i * 256);
    }
    size_t num_ones = static_cast<size_t>(msg_cnt * batch * p);
    std::vector<int> flat_vector(msg_cnt * batch, 0);
    std::fill(flat_vector.begin(), flat_vector.begin() + num_ones, 1);
    std::shuffle(flat_vector.begin(), flat_vector.end(), gen);
    // random mask with the specified sparsity in the plaintext
    std::deque<std::vector<double>> msg_m(msg_cnt, std::vector<double>(batch));
    for (size_t i = 0; i < msg_cnt; ++i) {
        for (size_t j = 0; j < batch; ++j) {
            msg_m[i][j] = flat_vector[i * batch + j];
        }
    }

    // check the pruning mask
    // for (const auto& row : msg_m) {
    //     for (const auto& val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // sort the pruning mask by sparsity (along with the corresponding ciphertext)
    auto countOnes = [](const std::vector<double>& v) {
        return std::count(v.begin(), v.end(), 1.0);
    };
    std::sort(msg_m.begin(), msg_m.end(), [&](const std::vector<double>& a, const std::vector<double>& b) {
        return countOnes(a) > countOnes(b);
    });
    // Print the sorted 2D vector
    // std::cout << "sorted by sparsity" << std::endl;
    // for (const auto& row : msg_m) {
    //     for (const auto& val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    auto keyPair = cc->KeyGen();
    std::cout << "Generating evaluation key for homomorphic multiplication...";
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalRotateKeyGen(keyPair.secretKey, rot_loc);
    std::cout << "Completed." << std::endl;

    // encrypt all p and y (during private training, p and y are always encrypted)
    std::vector<Ciphertext<DCRTPoly>> ct_d;
    Plaintext pt_temp;
    ct_d.reserve(msg_cnt);
    for (uint32_t i = 0; i < msg_cnt; i++){
      pt_temp = cc->MakeCKKSPackedPlaintext(msg_d[i]);
      ct_d.push_back(cc->Encrypt(keyPair.publicKey, pt_temp));
      break;
    }

    // compute ciphertext wise merging on the ciphertexts
    TIC(t1);
    while (msg_m.size() > 1){
        // Check and pop front if it's full
        if (std::all_of(msg_m.front().begin(), msg_m.front().end(), [](double val) { return val == 1.0; })) {
            // std::cout << "Popping full front." << std::endl;
            msg_m.pop_front();
            continue;
        }
        // Check and pop back if it's empty (all zeros)
        if (std::all_of(msg_m.back().begin(), msg_m.back().end(), [](double val) { return val == 0.0; })) {
            // std::cout << "Popping empty back." << std::endl;
            msg_m.pop_back();
            continue;
        }

        // Transfer from back to front
        std::vector<double>& m_back = msg_m.back();
        std::vector<double>& m_front = msg_m.front();

        // Find the first 1 in m_back and first 0 in m_front
        auto back_first_one = std::find(m_back.begin(), m_back.end(), 1);
        auto front_first_zero = std::find(m_front.begin(), m_front.end(), 0);

        // Check if a merge is possible
        while (back_first_one != m_back.end() && front_first_zero != m_front.end()) {
            // TIC(t2);
            size_t back_index = std::distance(m_back.begin(), back_first_one);
            size_t front_index = std::distance(m_front.begin(), front_first_zero);

            // std::cout << "Deque size:" << msg_m.size() << std::endl;
            // std::cout << "Front vector: ";
            // for (double num : m_front) {
            //   std::cout << num << " ";
            // }
            // std::cout << std::endl;
            // std::cout << "Back vector: ";
            // for (double num : m_back) {
            //   std::cout << num << " ";
            // }
            // std::cout << std::endl;

            // Calculate the minimal rotation needed
            // positive means left rotate, negative means right rotate
            size_t rotation = (front_index <= back_index) ? (back_index - front_index) : (m_back.size() + back_index - front_index);
            // std::cout << "rot:" << rotation << std::endl;
            auto cRot = ct_d[0];
            if(rotation != 0){
              cRot = cc->EvalRotate(ct_d[0], rotation*256);
              rot_cnt += 1;
            }
            std::vector<double> temp_mask(batchSize, 0.0);
            for (size_t i = front_index*256; i < 256; i++){
              temp_mask[i] = 1.0;
            }
            cRot = cc->EvalMult(cRot, cc->MakeCKKSPackedPlaintext(temp_mask));
            ct_d[0] = cc->EvalAdd(ct_d[0], cRot);

            m_front[front_index] = 1;
            m_back[back_index] = 0;

            // Find the next 1 in m_back and next 0 in m_front
            back_first_one = std::find(m_back.begin(), m_back.end(), 1);
            front_first_zero = std::find(m_front.begin(), m_front.end(), 0);
            // timeEval = TOC(t2);
        }
    }
    timeAll = TOC(t1);

    // std::cout << "[CWP] one merge time: " << timeEval << " ms" << std::endl;
    std::cout << "[CWP] rotation number: " << rot_cnt << std::endl;
    std::cout << "[CWP] evaluation time: " << timeAll << " ms" << std::endl;

    return 0;
}
