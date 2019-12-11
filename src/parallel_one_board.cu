#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

#include "cube_notion.h"
#include "cube_generator.h"

__global__ bool dfs(double limit, uint8_t* cur, double g, int preturn){
    // judge whether we have found the solution
    bool reach_target = true;
    for (int i = 0;i < 8;i ++){
        if (cur[i] / 3 != i) reach_target = false;
    }
    for (int i = 8;i < 20 && reach_target;i ++){
        if (cur[i] / 2 - 4 != i) reach_target = false;
    }
    if (reach_target){
        return true;
    }

    // calc current heuristic
    double h = 0.0;
    for (int i = 8;i < 20;i ++){
        h += heuristic[cur[i] / 2 - 12][cur[i] % 2 + (i - 8) * 2];
    }
    h /= 4.0;
    if (g + h > limit){
        // no need to explore current node
        if (*min_ext_h < 0.0 || g + h < *min_ext_h){
            *min_ext_h = g + h;
        }
    }else{
        // need to explore the node
        uint8_t * tmp = new uint8_t[20];
        for (int i = 0;i < kNumTurns;i ++){
            if (preturn / 3 == i / 3){
                continue;
            }
            for (int j = 0;j < 20;j ++) tmp[j] = cur[j];
            TurnCube(tmp, i);
            if (dfs(limit, tmp, g + 1, i)){
                ans[++ ans[0]] = i;
                return true;
            }
        }
    }

    return false;
}

__global__ void iterative_deepening_astar(uint8_t* cur){
    double limit = 0.0;
    bool find = false;
    *min_ext_h = limit;
    ans[0] = 0;
    while (!find) {
        limit = *min_ext_h;
        *min_ext_h = -1;
        find = dfs(limit, cur, 0.0, -1);
        //cout << "limit = " << limit << endl;
    }
    cout << "solution sequence:" << endl;
    for (int i = ans[0];i; -- i){
        cout << turns_str[ans[i]] <<", ";
    }
    cout << endl;
}

int main(){
    // check the GPU status
    int* numofgpu = new int;
    cudaGetDeviceCount(numofgpu);

    std :: cout << "Number of GPUs on board" << numofgpu << std :: endl;

    // read data
    // freopen("data.txt","r",stdin);
    // string str;
    // getline(cin,str);

    // uint8_t* org_state = new uint8_t[CUBE_ARR_SIZE];
    // uint8_t* cur_state = new uint8_t[CUBE_ARR_SIZE];
    // uint8_t org_state[CUBE_ARR_SIZE];
    uint8_t cur_state[CUBE_ARR_SIZE];

    generate_cube(cur_state, 10);
    // read_state(org_state);

    // string scramble_seq;
    // getline(cin, scramble_seq);
    //cout << scramble_seq << endl;

    // read_state(cur_state);

    // construct_heuristic_table(org_state);

    // Add
    uint8_t *ans;
    double *min_ext_h;
    cudaError_t ret;
    // Allocate Unified Memory – accessible from CPU or GPU
    ret = cudaMalloc(&ans, 30);
    if (ret != cudaSuccess) {
      cout << "Failed to allocate ans" << endl;
      return 1;
    }
    ret = cudaMalloc(&min_ext_h, sizeof(double));
    if (ret != cudaSuccess) {
      cout << "Failed to allocate min_ext_h" << endl;
      return 1;
    }
    // ret = cudaMemcpy(gA, (void *)cA, n * n * sizeof(double), cudaMemcpyHostToDevice);
    // /* cudaStatus = cudaMemcpy(host_data, dev_data, N*sizeof(float), cudaMemcpyDeviceToHost); */
    // if (ret != cudaSuccess) {
    //   cout << "Failed to copy A to GPU" << endl;
    //   return 1;
    // }

    cout << "Start timer" << endl;
    cudaEvent_t stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* time_t op = time(NULL); */
    iterative_deepening_astar(cur_state);
    /* time_t ed = time(NULL); */

    // Do work

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "Timer stopped" << endl;
    cout << "Time consumed: " << elapsed_time << endl;
    /* cout << "Time consumed: " << ed - op << "sec" << endl; */

    // Check correctness
    for (int i = ans[0];i; -- i){
        TurnCube(cur_state, ans[i]);
    }
    PrintCube(cur_state, CUBE_ARR_SIZE);
    
    // Free memory here

    return 0;
}
