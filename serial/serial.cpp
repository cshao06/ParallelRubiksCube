#include <cstdio>
#include <iostream>
#include <cstring>
#include <cctype>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <inttypes.h>
#include <ctime>

#include "cube_notions.h"
#include "cube_generator.h"

using namespace std;

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

    /*
    for (int i = 0;i < 20;i ++){
        cout << +org_state[i] << " ";
    }
    cout << endl;
    TurnCube(org_state, 5);
    for (int i = 0;i < 20;i ++){
        cout << +org_state[i] << " ";
    }
    cout << endl;
    */

    uint8_t* tmp = new uint8_t[20];
    cnt[0] = 0;
    while (l < r){
        for (int i = 0;i < kNumTurns;i ++){
            memcpy(tmp, que[l], sizeof(uint8_t) * 20);
            TurnCube(tmp, i);
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

int ans[30];
double min_ext_h;
bool dfs(double limit, uint8_t* cur, double g, int preturn){
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
        if (min_ext_h < 0.0 || g + h < min_ext_h){
            min_ext_h = g + h;
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
void iterative_deepening_astar(uint8_t* cur){
    double limit = 0.0;
    bool find = false;
    min_ext_h = limit;
    ans[0] = 0;
    while (!find) {
        limit = min_ext_h;
        min_ext_h = -1;
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

    /*
    for (int i = 0;i < 20;i ++){
        cout << org_state[i] << " ";
    }
    cout << endl;

    for (int i = 0;i < 20;i ++){
        cout << cur_state[i] << " ";
    }
    cout << endl;
    */

    construct_heuristic_table(org_state);
    /*
    for (int i = 0;i < 12;i ++) {
        for (int j = 0;j < 24;j ++){
            printf("%d, ",heuristic[i][j]);
        }
        printf("\n");
    }
    */
    clock_t op = clock();
    iterative_deepening_astar(cur_state);
    clock_t ed = clock();
    cout << "Time consumed: " << ed - op << "ms" << endl;

    for (int i = ans[0];i; -- i){
        TurnCube(cur_state, ans[i]);
    }
    for (int i = 0;i < 20;i ++){
        cout << +cur_state[i] << " ";
    }
    cout << endl;

    return 0;
}
