#include <cstdint>
#include <iostream>
#include <iomanip>
#include "cube_notions.h"

using namespace std;

const char *turns_str[kNumTurns] = {
  "U", "U'", "U2",
  "D", "D'", "D2",
  "F", "F'", "F2",
  "B", "B'", "B2",
  "L", "L'", "L2",
  "R", "R'", "R2",
};

const uint8_t positions_on_face[kNumFaces][NUM_POSITIONS_PER_FACE] = {
  [kU] = {kURF, kUFL, kULB, kUBR, kUF, kUL, kUB, kUR},
  [kD] = {kDLF, kDFR, kDRB, kDBL, kDF, kDR, kDB, kDL},
  [kF] = {kUFL, kURF, kDFR, kDLF, kUF, kFR, kDF, kFL},
  [kB] = {kUBR, kULB, kDBL, kDRB, kUB, kBL, kDB, kBR},
  [kL] = {kULB, kUFL, kDLF, kDBL, kUL, kFL, kDL, kBL},
  [kR] = {kURF, kUBR, kDRB, kDFR, kUR, kBR, kDR, kFR},
};

const uint8_t turn_position[kNumTurns][NUM_POSITIONS_PER_FACE] = {
  [kU_] = {kUFL, kULB, kUBR, kURF, kUL, kUB, kUR, kUF},
  [kUp] = {kUBR, kURF, kUFL, kULB, kUR, kUF, kUL, kUB},
  [kU2] = {kULB, kUBR, kURF, kUFL, kUB, kUR, kUF, kUL},
  [kD_] = {kDFR, kDRB, kDBL, kDLF, kDR, kDB, kDL, kDF},
  [kDp] = {kDBL, kDLF, kDFR, kDRB, kDL, kDF, kDR, kDB},
  [kD2] = {kDRB, kDBL, kDLF, kDFR, kDB, kDL, kDF, kDR},
  [kF_] = {kURF, kDFR, kDLF, kUFL, kFR, kDF, kFL, kUF},
  [kFp] = {kDLF, kUFL, kURF, kDFR, kFL, kUF, kFR, kDF},
  [kF2] = {kDFR, kDLF, kUFL, kURF, kDF, kFL, kUF, kFR},
  [kB_] = {kULB, kDBL, kDRB, kUBR, kBL, kDB, kBR, kUB},
  [kBp] = {kDRB, kUBR, kULB, kDBL, kBR, kUB, kBL, kDB},
  [kB2] = {kDBL, kDRB, kUBR, kULB, kDB, kBR, kUB, kBL},
  [kL_] = {kUFL, kDLF, kDBL, kULB, kFL, kDL, kBL, kUL},
  [kLp] = {kDBL, kULB, kUFL, kDLF, kBL, kUL, kFL, kDL},
  [kL2] = {kDLF, kDBL, kULB, kUFL, kDL, kBL, kUL, kFL},
  [kR_] = {kUBR, kDRB, kDFR, kURF, kBR, kDR, kFR, kUR},
  [kRp] = {kDFR, kURF, kUBR, kDRB, kFR, kUR, kBR, kDR},
  [kR2] = {kDRB, kDFR, kURF, kUBR, kDR, kFR, kUR, kBR},
};

const uint8_t turn_orientation[kNumTurns][NUM_POSITIONS_PER_FACE] = {
  [kU_] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kUp] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kU2] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kD_] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kDp] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kD2] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kF_] = {2, 1, 2, 1, 1, 1, 1, 1},
  [kFp] = {2, 1, 2, 1, 1, 1, 1, 1},
  [kF2] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kB_] = {2, 1, 2, 1, 1, 1, 1, 1},
  [kBp] = {2, 1, 2, 1, 1, 1, 1, 1},
  [kB2] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kL_] = {2, 1, 2, 1, 0, 0, 0, 0},
  [kLp] = {2, 1, 2, 1, 0, 0, 0, 0},
  [kL2] = {0, 0, 0, 0, 0, 0, 0, 0},
  [kR_] = {2, 1, 2, 1, 0, 0, 0, 0},
  [kRp] = {2, 1, 2, 1, 0, 0, 0, 0},
  [kR2] = {0, 0, 0, 0, 0, 0, 0, 0},
};

const uint8_t edge_manhattan_dist[12][24] = {
  {0, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2},
  {1, 2, 0, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 2, 2, 2},
  {1, 3, 1, 2, 0, 3, 1, 2, 2, 3, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1},
  {1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1, 2},
  {1, 3, 2, 2, 2, 3, 2, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2},
  {2, 2, 2, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 2, 2, 2, 1, 2},
  {2, 3, 2, 2, 1, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1},
  {2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 2, 1, 2, 2, 2},
  {2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 3, 1, 3, 2, 3, 1, 3},
  {2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 3, 0, 3, 1, 3, 2, 3},
  {2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1, 3, 0, 3, 1, 3},
  {2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 3, 0, 3},
};

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

void PrintCube(uint8_t *cube, uint8_t size) {
  cout << "[ ";
  for (uint8_t i = 0; i < size; i++) {
    cout << setw(2) << +cube[i] << ", ";
  }
  cout << "]" << endl;
}

