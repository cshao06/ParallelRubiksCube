#include <stdio.h>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

#include "cube_notions.h"
#include "cube_generator.h"

__global__ void gpu_search(uint8_t* data, uint8_t* past_step, int* global_best_step, uint8_t* d_heuristic){
    __shared__ uint8_t heuristic[12][24];
    if (threadIdx.x == 0){
        for (int i = 0, k = 0;i < 12;i ++)
            for (int j = 0;j < 24;j ++){
                heuristic[i][j] = d_heuristic[k ++];
            }
    }
    __syncthreads();
    int global_index = (threadIdx.x + blockIdx.x * blockDim.x) * 20;
    uint8_t sk[20][20];
    int g[20];
    bool visit[20];
    for (int i = 0;i < 20;i ++){
        visit[i] = false;
    }
    int cur_arc[20];
    int dep = 0;
    g[0] = past_step[threadIdx.x + blockIdx.x * blockDim.x];
    for (int i = 0;i < 20;i ++){
        sk[0][i] = data[global_index + i];
    }

    //maximum number of steps need to solve the rubick
    uint8_t upper_bound = 11;

    //record preturn, avoid continuous same rotate
    int preturn[20];

    //IDA*
    double limit = g[0];
    bool find_ans = false;
    double min_ext_h = limit;
    while (!find_ans){
        if (min_ext_h < 0){
            break;
        }
        limit = min_ext_h;
        /*
        if (limit > upper_bound){
            break;
        }
        */
        min_ext_h = -1;
        //need to check the best answer when having explored enough nodes
        int num_of_ext_nodes = 0;
        dep = 0;
        for (int i = 0;i < 20;i ++){
            visit[i] = false;
        }
        //break;
        while (dep >= 0){
            // visit = true -> check cur_arc, otherwise do some calculation first;
            if (visit[dep] == false){
                visit[dep] = true;

                if (++ num_of_ext_nodes == 1000){
                    uint8_t tmp = *global_best_step;
                    if (tmp < upper_bound){
                        upper_bound = tmp;
                    }
                    num_of_ext_nodes = 0;
                }
                // judge whether we have found the solution
                bool reach_target = true;
                for (int i = 0;i < 8;i ++){
                    if (sk[dep][i] != i * 3) reach_target = false;
                }
                for (int i = 8;i < 20 && reach_target;i ++){
                    if (sk[dep][i] != (i + 4) * 2) reach_target = false;
                }
                if (reach_target){
                    // find answer
                    find_ans = true;
                    atomicMin(global_best_step, g[dep]);
                    break;
                }

                // calc current heuristic
                double h = 0.0;
                for (int i = 8;i < 20;i ++){
                    h += heuristic[sk[dep][i] / 2 - 12][sk[dep][i] % 2 + (i - 8) * 2];
                }
                h /= 4.0;
                if (g[dep] + h > upper_bound){
                    visit[dep --] = false;
                    continue;
                }
                if (g[dep] + h > limit){
                    if (min_ext_h < 0.0 || g[dep] + h < min_ext_h){
                        min_ext_h = g[dep] + h;
                    }
                    visit[dep --] = false;
                    continue;
                }else{
                    for (int i = 0;i < kNumTurns;i ++){
                        if (dep > 0 && preturn[dep] / 3 == i / 3){
                            continue;
                        }
                        for (int j = 0;j < 20;j ++){
                            sk[dep + 1][j] = sk[dep][j];
                        }

                        TurnCube(sk[dep + 1], i);
                        preturn[dep + 1] = i;
                        g[dep + 1] = g[dep] + 1;
                        cur_arc[dep] = i;
                        ++ dep;
                        break;
                    }
                }
            }else{
                bool find_new_leaf = false;
                for (++ cur_arc[dep]; cur_arc[dep] < kNumTurns;++ cur_arc[dep]){
                    if (dep > 0 && preturn[dep] / 3 == cur_arc[dep] / 3){
                        continue;
                    }
                    for (int j = 0;j < 20;j ++){
                        sk[dep + 1][j] = sk[dep][j];
                    }

                    TurnCube(sk[dep + 1], cur_arc[dep]);
                    preturn[dep + 1] = cur_arc[dep];
                    g[dep + 1] = g[dep] + 1;
                    ++ dep;
                    find_new_leaf = true;
                    break;
                }
                if (!find_new_leaf) visit[dep --] = false;
            }
        }
    }
}

int heuristic[12][24];

void construct_heuristic_table(uint8_t* org_state){
    memset(heuristic, -1, sizeof(heuristic));
    for (int i = 0;i < 12;i ++) heuristic[i][i << 1] = 0;

    int mx = 18 * 18 * 18;

    uint8_t ** que = new uint8_t* [mx + 2];
    uint8_t * cnt = new uint8_t[mx + 2];

    for (int i = 0;i < mx;i ++) que[i] = new uint8_t[20];
    for (int i = 0;i < 20;i ++) que[0][i] = org_state[i];
    int l = 0, r = 1;

    uint8_t* tmp = new uint8_t[20];
    cnt[0] = 0;
    while (l < r){
        for (int i = 0;i < kNumTurns;i ++){
            memcpy(tmp, que[l], sizeof(uint8_t) * 20);
            TurnCubeCPU(tmp, i);
            bool have_new = false;
            for (int j = 8;j < 20;j ++){
                if (heuristic[tmp[j] / 2 - 12][tmp[j] % 2 + (j - 8) * 2] == -1){
                    have_new = true;
                    heuristic[tmp[j] / 2 - 12][tmp[j] % 2 + (j - 8) * 2] = cnt[l] + 1;
                }
            }
            if (have_new){
                cnt[r] = cnt[l] + 1;
                memcpy(que[r], tmp, sizeof(uint8_t) * 20);
                ++ r;
            }
        }
        ++ l;
    }
}

const int gridsize = 512;
const int blocksize = 1024;
uint8_t que[gridsize * blocksize * 2 + 5][20];
uint8_t que_flat[gridsize * blocksize * 2 * 20];
uint8_t cnt[gridsize * blocksize * 2 + 5];
uint8_t cnt_flat[gridsize * blocksize * 2];
uint8_t heuristic_flat[12 * 24];

void generate_subproblems(uint8_t* cur, int* numofgpu){
    int numofdev = *numofgpu;
    memcpy(que[0], cur, sizeof(uint8_t) * 20);
    int mod = gridsize * blocksize * numofdev + 1;
    int l = 0, r = 1, numofsub = 1;
    cnt[0] = 0;
    while (true){
        for (int i = 0;i < kNumTurns;i ++){
            memcpy(que[r], que[l], sizeof(uint8_t) * 20);
            TurnCubeCPU(que[r], i);
            cnt[r] = cnt[l] + 1;
            r = (r + 1) % mod;
            if (++ numofsub == mod - 1){
                break;
            }
        }
        if (numofsub == mod - 1){
            break;
        }
        l = (l + 1) % mod;
        -- numofsub;
    }

    for (int i = l,j = 0, w = 0;i != r;i = (i + 1) % mod){
        for (int k = 0;k < 20;k ++){
            que_flat[j ++] = que[i][k];
        }
        cnt_flat[w ++] = cnt[i];
    }
    for (int i = 0;i < 12 * 24;i ++){
        heuristic_flat[i] = heuristic[i / 24][i % 24];
    }

    uint8_t* d[2];
    uint8_t* d_step[2];
    int* best_step[2];
    uint8_t* d_heuristic[2];
    for (int dev_id = 0;dev_id < numofdev;dev_id ++){
        cudaSetDevice(dev_id);

        cudaMalloc((void **)&d[dev_id], gridsize * blocksize * 20 * sizeof(uint8_t));
        cudaMemcpy(d[dev_id], que_flat + dev_id * gridsize * blocksize * 20, gridsize * blocksize * 20 * sizeof(uint8_t), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&d_step[dev_id], gridsize * blocksize * sizeof(uint8_t));
        cudaMemcpy(d_step[dev_id], cnt_flat + dev_id * gridsize * blocksize, gridsize * blocksize * sizeof(uint8_t), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&best_step[dev_id], sizeof(int));
        cudaMemset(best_step[dev_id], 11, sizeof(int));

        cudaMalloc((void **)&d_heuristic[dev_id], 12 * 24 * sizeof(uint8_t));
        cudaMemcpy(d_heuristic[dev_id], heuristic_flat, 12 * 24 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    dim3 grid_dim = dim3(gridsize, 1, 1);
    dim3 block_dim = dim3(blocksize, 1, 1);

    std :: cout << "Start timer" << std :: endl;
    cudaEvent_t stop, start;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int dev_id = 0;dev_id < numofdev; dev_id ++){
        cudaSetDevice(dev_id);
        gpu_search<<<grid_dim, block_dim>>>(d[dev_id], d_step[dev_id], best_step[dev_id], d_heuristic[dev_id]);
    }

    // Wait for GPU to finish before accessing on host
    for (int dev_id = 0;dev_id < *numofgpu; dev_id ++){
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "Timer stopped" << endl;
    cout << "Time consumed: " << elapsed_time << endl;

    uint8_t* final_answer[2];
    for (int i = 0;i < 2;i ++){
        final_answer[i] = new uint8_t;
    }

    for (int dev_id = 0;dev_id < numofdev; dev_id ++){
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
        cudaMemcpy(final_answer[dev_id], best_step[dev_id], sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }

    uint8_t optimal_step = *final_answer[0];
    for (int dev_id = 1; dev_id < numofdev; dev_id ++){
        if ((*final_answer[dev_id]) < optimal_step){
            optimal_step = *final_answer[dev_id];
        }
    }
    std :: cout << "Optimal solution to recover the cube: " << +optimal_step << std :: endl;
}

int main(){
    // check the GPU status
    int* numofgpu = new int;
    cudaGetDeviceCount(numofgpu);
    std :: cout << "Number of GPUs on board:\n" << *numofgpu << std :: endl;

    uint8_t cur_state[CUBE_ARR_SIZE];
    generate_cube(cur_state, 8);

    uint8_t* org_state = new uint8_t[20];
    for (int i = 0;i < 8;i ++) org_state[i] = i * 3;
    for (int i = 8;i < 20;i ++) org_state[i] = 24 + (i - 8) * 2;
    construct_heuristic_table(org_state);

    generate_subproblems(cur_state, numofgpu);
    
    // Free memory here
    for (int dev_id = 0; dev_id < *numofgpu; dev_id ++){
        cudaSetDevice(dev_id);
        cudaDeviceReset();
    }

    return 0;
}
