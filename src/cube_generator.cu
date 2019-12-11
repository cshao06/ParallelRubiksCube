#include <iostream>
#include <iomanip>
#include <time.h>
#include "cube_generator.h"
// #include "cube_notions_cpu.h"
#include "cube_notions.h"

using namespace std;

// The original state
const uint8_t solved_cube[CUBE_ARR_SIZE] = {kUrf, kUfl, kUlb, kUbr, kDlf, kDfr, kDrb, kDbl, kUf, kUl, kUb, kUr, kDf, kDr, kDb, kDl, kFr, kFl, kBl, kBr};

void read_state(uint8_t* state){
    string str;
    getline(cin,str);
    //cout << str << endl;
    int len = str.length();
    int num = -1, id = 0;
    for (int i = 0;i < len;i ++){
        if (!isdigit(str[i])){
            // judge whether need to update state
            if (num != -1){
                state[id ++] = uint8_t (num);
                num = -1;
            }
        }else{
            // update num
            if (num == -1){
                num = str[i] - '0';
            }else{
                num = num * 10 + str[i] - '0';
            }
        }
    }
}


void PrintCube(uint8_t *cube, uint8_t size) {
  cout << "[ ";
  for (uint8_t i = 0; i < size; i++) {
    cout << setw(2) << +cube[i] << ", ";
  }
  cout << "]" << endl;
}

void generate_cube(uint8_t *cube, uint8_t num_turns) {
  memcpy(cube, solved_cube, CUBE_ARR_SIZE);
  uint8_t *scramble = (uint8_t *)malloc(num_turns);

  srand(time(NULL));

  cout << "Original state: " << endl;
  PrintCube(cube, CUBE_ARR_SIZE);
  scramble[0] = rand() % kNumTurns;
  TurnCubeCPU(cube, scramble[0]);
  for (uint8_t i = 1; i < num_turns; i++) {
    do {
      scramble[i] = rand() % kNumTurns;
    } while (scramble[i - 1] / 3 == scramble[i] / 3);
    // cout << +scramble[i] << endl;
    TurnCubeCPU(cube, scramble[i]);
    // PrintCube(cube, sizeof(cube));
  }

  for (uint8_t i = 0; i < num_turns; i++) {
    // cout << setw(2) << turns_str[i] << " ";
    cout << turns_str[scramble[i]] << " ";
  }
  cout << endl;
  PrintCube(cube, CUBE_ARR_SIZE);

  free(scramble);
}
