#include <iostream>
#include <iomanip>
#include <time.h>
#include "cube_notions.h"

using namespace std;

// The original state
uint8_t cube[8 + 12] = {kUrf, kUfl, kUlb, kUbr, kDlf, kDfr, kDrb, kDbl, kUf, kUl, kUb, kUr, kDf, kDr, kDb, kDl, kFr, kFl, kBl, kBr};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: ./cube_generator num_turns" << endl;
    return 1;
  }
  uint8_t num_turns = atoi(argv[1]);
  uint8_t *scramble = (uint8_t *)malloc(num_turns);

  srand(time(NULL));

  cout << "Original state: " << endl;
  PrintCube(cube, sizeof(cube));
  scramble[0] = rand() % kNumTurns;
  // scramble[0] = kU2;
  TurnCube(cube, scramble[0]);
  for (uint8_t i = 1; i < num_turns; i++) {
    do {
      scramble[i] = rand() % kNumTurns;
    } while (scramble[i - 1] / 3 == scramble[i] / 3);
    // cout << +scramble[i] << endl;
    TurnCube(cube, scramble[i]);
    // PrintCube(cube, sizeof(cube));
  }

  for (uint8_t i = 0; i < num_turns; i++) {
    // cout << setw(2) << turns_str[i] << " ";
    cout << turns_str[scramble[i]] << " ";
  }
  cout << endl;
  PrintCube(cube, sizeof(cube));

  free(scramble);
}
