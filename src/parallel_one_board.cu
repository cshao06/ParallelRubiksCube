#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

int main(){
    // check the GPU status
    int* numofgpu = new int;
    cudaGetDeviceCount(numofgpu);

    std :: cout << "Number of GPUs on board" << numofgpu << std :: endl;

    // read data
    freopen("data.txt","r",stdin);
    string str;
    getline(cin,str);

    uint8_t* org_state = new uint8_t[20];
    uint8_t* cur_state = new uint8_t[20];

    read_state(org_state);

    string scramble_seq;
    getline(cin, scramble_seq);
    //cout << scramble_seq << endl;

    read_state(cur_state);

    construct_heuristic_table(org_state);

    time_t op = time(NULL);
    iterative_deepening_astar(cur_state);
    time_t ed = time(NULL);
    cout << "Time consumed: " << ed - op << "sec" << endl;

    for (int i = ans[0];i; -- i){
        TurnCube(cur_state, ans[i]);
    }
    for (int i = 0;i < 20;i ++){
        cout << +cur_state[i] << " ";
    }
    cout << endl;

    return 0;
}