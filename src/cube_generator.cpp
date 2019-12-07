#include <iostream>
#include <iomanip>
#include <time.h>
#include "cube_notions.h"

using namespace std;

// The original state
uint8_t cube[8 + 12] = {kUrf, kUfl, kUlb, kUbr, kDlf, kDfr, kDrb, kDbl, kUf, kUl, kUb, kUr, kDf, kDr, kDb, kDl, kFr, kFl, kBl, kBr};

void PrintCube(uint8_t *cube, uint8_t size) {
  cout << "[ ";
  for (uint8_t i = 0; i < size; i++) {
    cout << setw(2) << +cube[i] << ", ";
  }
  cout << "]" << endl;
}

void TurnCube(uint8_t *cube, uint8_t turn) {
  const uint8_t *positions = positions_on_face[turn / 3];
  uint8_t tmp[8];
  for (uint8_t i = 0; i < 8; i++) {
    tmp[i] = cube[positions[i]];
  }
  // For each corner position that this turn will affect
  for (uint8_t i = 0; i < NUM_POSITIONS_PER_FACE / 2; i++) {
    uint8_t new_value = tmp[i] + turn_orientation[turn][i];
    if (tmp[i] / 3 != new_value / 3) {
      new_value -= 3;
    }
    cube[turn_position[turn][i]] = new_value;
  }
  // For each edge position that this turn will affect
  for (uint8_t i = NUM_POSITIONS_PER_FACE / 2; i < NUM_POSITIONS_PER_FACE; i++) {
    uint8_t new_value = tmp[i] + turn_orientation[turn][i];
    if (tmp[i] / 2 != new_value / 2) {
      new_value -= 2;
    }
    cube[turn_position[turn][i]] = new_value;
  }
}

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
