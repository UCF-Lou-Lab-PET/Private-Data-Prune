#define PROFILE  // turns on the reporting of timing results

#include "openfhe.h"
#include "ArgMapping.h"
#include <ctime>
#include <random>

using namespace lbcrypto;
/*
input: the prediction vector P, the ground truth label Y
output: the importance score.
other parameters to test this component:
  cls: number of class
  ctn: number of ciphertexts
*/
int main(int argc, char* argv[]) {

    /************* Argument Parsing  ************/
    /********************************************/
    uint32_t cls = 10;
    uint32_t ctn = 100;
    ArgMapping amap;
    amap.arg("cls", cls, "the number of class");
    amap.arg("ctn", ctn, "the number of ciphertexts");
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
    std::cout << "Dataset: number of class            " << cls << std::endl;
    std::cout << "Dataset: number of ciphertexts      " << ctn << std::endl;

    // coeff for the approximation poly.
    std::vector<double> coefficients1({
      -5.30489756589578e-48, // x^0
      8.83133072022416,      // x^1
      2.15841891006552e-46,  // x^2
      -4.64575039895512e1,   // x^3
      0,                     // x^4
      8.30282234720408e1,    // x^5
      0,                     // x^6
      -4.49928477828070e1    // x^7
    });

    std::vector<double> coefficients2({
        0,                    // x^0
        3.94881885083263,     // x^1
        0,                    // x^2
        -1.29103010992282e1,  // x^3
        0,                    // x^4
        2.80865362174658e1,   // x^5
        0,                    // x^6
        -3.55969148965137e1,  // x^7
        0,                    // x^8
        2.65159370881337e1,   // x^9
        0,                    // x^10
        -1.14184889368449e1,  // x^11
        0,                    // x^12
        2.62558443881334,     // x^13
        0,                    // x^14
        -2.49172299998642e-1  // x^15
    });

    uint32_t msg_cnt = ctn;
    uint32_t msg_length = cls;
    TimeVar t;
    double timeEval(0.0), timeAll(0.0);
    std::random_device rd;  // Random device
    std::mt19937 gen(rd()); // Mersenne Twister generator
    std::uniform_real_distribution<> dis1(-1.0, 1.0);
    std::uniform_int_distribution<> dis2(0, msg_length - 1);

    std::vector<std::vector<double>> msg_p(msg_cnt, std::vector<double>(msg_length));
    for (uint32_t i = 0; i < msg_cnt; ++i) {
        for (size_t j = 0; j < msg_length; ++j) {
            msg_p[i][j] = dis1(gen);
        }
    }
    std::vector<std::vector<double>> msg_y(msg_cnt, std::vector<double>(msg_length));
    for (uint32_t i = 0; i < msg_cnt; ++i) {
        int index_t = dis2(gen);
        msg_y[i][index_t] = 1.0;
    }
    // size_t encodedLength = msg_length;

    auto keyPair = cc->KeyGen();
    std::cout << "Generating evaluation key for homomorphic multiplication...";
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalRotateKeyGen(keyPair.secretKey, {1, 2, 4, 8, 16});
    std::cout << "Completed." << std::endl;

    // encrypt all p and y (during private training, p and y are always encrypted)
    std::vector<Ciphertext<DCRTPoly>> ct_p;
    std::vector<Ciphertext<DCRTPoly>> ct_y;
    Plaintext pt_temp;
    ct_p.reserve(msg_cnt);
    ct_y.reserve(msg_cnt);
    for (uint32_t i = 0; i < msg_cnt; i++){
      pt_temp = cc->MakeCKKSPackedPlaintext(msg_p[i]);
      ct_p.push_back(cc->Encrypt(keyPair.publicKey, pt_temp));
      pt_temp = cc->MakeCKKSPackedPlaintext(msg_y[i]);
      ct_y.push_back(cc->Encrypt(keyPair.publicKey, pt_temp));
      break;
    }

    // compute hefs on each ciphertext
    for (uint32_t i = 0; i < msg_cnt; i++){
      size_t id = 0;
      TIC(t);
      // compute max(p-y, y-p), ref: https://eprint.iacr.org/2020/834
      // let u=p-y and v=y-p; max = ((u+v) + (u-v)sign(u-v)) / 2 = (p-y)sign(p-y)
      auto cDiff = cc->EvalSub(ct_p[id], ct_y[id]);
      auto res1 = cc->EvalPoly(cDiff, coefficients1);
      auto res2 = cc->EvalPoly(res1, coefficients2);
      auto res3 = cc->EvalMult(cDiff, res2);
      // rotate sum
      for (uint32_t j=1; j<cls; j*=2){
        // std::cout<<"rotate" << std::endl;
        res1 = cc->EvalRotate(res3, j);
        res2 = cc->EvalAdd(res1, res3);
        res3 = res2;
      }
      timeEval = TOC(t);
      // std::cout <<"one HEFS time" << timeEval << std::endl;
      timeAll += timeEval;
    }

    std::cout << "HEFS evaluation time: " << timeAll << " ms" << std::endl;

    return 0;
}
