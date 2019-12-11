#include <stdint.h>
#include "cube_notions.h"

const char *turns_str[kNumTurns] = {
  "U", "U'", "U2",
  "D", "D'", "D2",
  "F", "F'", "F2",
  "B", "B'", "B2",
  "L", "L'", "L2",
  "R", "R'", "R2",
};

// #ifdef __CUDA_ARCH__
__constant__ const uint8_t positions_on_face[kNumFaces][NUM_POSITIONS_PER_FACE] = {
  {kURF, kUFL, kULB, kUBR, kUF, kUL, kUB, kUR},
  {kDLF, kDFR, kDRB, kDBL, kDF, kDR, kDB, kDL},
  {kUFL, kURF, kDFR, kDLF, kUF, kFR, kDF, kFL},
  {kUBR, kULB, kDBL, kDRB, kUB, kBL, kDB, kBR},
  {kULB, kUFL, kDLF, kDBL, kUL, kFL, kDL, kBL},
  {kURF, kUBR, kDRB, kDFR, kUR, kBR, kDR, kFR},
};
// #endif

const uint8_t positions_on_face_cpu[kNumFaces][NUM_POSITIONS_PER_FACE] = {
        {kURF, kUFL, kULB, kUBR, kUF, kUL, kUB, kUR},
        {kDLF, kDFR, kDRB, kDBL, kDF, kDR, kDB, kDL},
        {kUFL, kURF, kDFR, kDLF, kUF, kFR, kDF, kFL},
        {kUBR, kULB, kDBL, kDRB, kUB, kBL, kDB, kBR},
        {kULB, kUFL, kDLF, kDBL, kUL, kFL, kDL, kBL},
        {kURF, kUBR, kDRB, kDFR, kUR, kBR, kDR, kFR},
};

// #ifdef __CUDA_ARCH__
__constant__ const uint8_t turn_position[kNumTurns][NUM_POSITIONS_PER_FACE] = {
  {kUFL, kULB, kUBR, kURF, kUL, kUB, kUR, kUF},
  {kUBR, kURF, kUFL, kULB, kUR, kUF, kUL, kUB},
  {kULB, kUBR, kURF, kUFL, kUB, kUR, kUF, kUL},
  {kDFR, kDRB, kDBL, kDLF, kDR, kDB, kDL, kDF},
  {kDBL, kDLF, kDFR, kDRB, kDL, kDF, kDR, kDB},
  {kDRB, kDBL, kDLF, kDFR, kDB, kDL, kDF, kDR},
  {kURF, kDFR, kDLF, kUFL, kFR, kDF, kFL, kUF},
  {kDLF, kUFL, kURF, kDFR, kFL, kUF, kFR, kDF},
  {kDFR, kDLF, kUFL, kURF, kDF, kFL, kUF, kFR},
  {kULB, kDBL, kDRB, kUBR, kBL, kDB, kBR, kUB},
  {kDRB, kUBR, kULB, kDBL, kBR, kUB, kBL, kDB},
  {kDBL, kDRB, kUBR, kULB, kDB, kBR, kUB, kBL},
  {kUFL, kDLF, kDBL, kULB, kFL, kDL, kBL, kUL},
  {kDBL, kULB, kUFL, kDLF, kBL, kUL, kFL, kDL},
  {kDLF, kDBL, kULB, kUFL, kDL, kBL, kUL, kFL},
  {kUBR, kDRB, kDFR, kURF, kBR, kDR, kFR, kUR},
  {kDFR, kURF, kUBR, kDRB, kFR, kUR, kBR, kDR},
  {kDRB, kDFR, kURF, kUBR, kDR, kFR, kUR, kBR},
};
// #endif

const uint8_t turn_position_cpu[kNumTurns][NUM_POSITIONS_PER_FACE] = {
  {kUFL, kULB, kUBR, kURF, kUL, kUB, kUR, kUF},
  {kUBR, kURF, kUFL, kULB, kUR, kUF, kUL, kUB},
  {kULB, kUBR, kURF, kUFL, kUB, kUR, kUF, kUL},
  {kDFR, kDRB, kDBL, kDLF, kDR, kDB, kDL, kDF},
  {kDBL, kDLF, kDFR, kDRB, kDL, kDF, kDR, kDB},
  {kDRB, kDBL, kDLF, kDFR, kDB, kDL, kDF, kDR},
  {kURF, kDFR, kDLF, kUFL, kFR, kDF, kFL, kUF},
  {kDLF, kUFL, kURF, kDFR, kFL, kUF, kFR, kDF},
  {kDFR, kDLF, kUFL, kURF, kDF, kFL, kUF, kFR},
  {kULB, kDBL, kDRB, kUBR, kBL, kDB, kBR, kUB},
  {kDRB, kUBR, kULB, kDBL, kBR, kUB, kBL, kDB},
  {kDBL, kDRB, kUBR, kULB, kDB, kBR, kUB, kBL},
  {kUFL, kDLF, kDBL, kULB, kFL, kDL, kBL, kUL},
  {kDBL, kULB, kUFL, kDLF, kBL, kUL, kFL, kDL},
  {kDLF, kDBL, kULB, kUFL, kDL, kBL, kUL, kFL},
  {kUBR, kDRB, kDFR, kURF, kBR, kDR, kFR, kUR},
  {kDFR, kURF, kUBR, kDRB, kFR, kUR, kBR, kDR},
  {kDRB, kDFR, kURF, kUBR, kDR, kFR, kUR, kBR},
};

// #ifdef __CUDA_ARCH__
__constant__ const uint8_t turn_orientation[kNumTurns][NUM_POSITIONS_PER_FACE] = {
   {0, 0, 0, 0, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {2, 1, 2, 1, 1, 1, 1, 1},
   {2, 1, 2, 1, 1, 1, 1, 1},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {2, 1, 2, 1, 1, 1, 1, 1},
   {2, 1, 2, 1, 1, 1, 1, 1},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {2, 1, 2, 1, 0, 0, 0, 0},
   {2, 1, 2, 1, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
   {2, 1, 2, 1, 0, 0, 0, 0},
   {2, 1, 2, 1, 0, 0, 0, 0},
   {0, 0, 0, 0, 0, 0, 0, 0},
};
// #endif

const uint8_t turn_orientation_cpu[kNumTurns][NUM_POSITIONS_PER_FACE] = {
 {0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {2, 1, 2, 1, 1, 1, 1, 1},
 {2, 1, 2, 1, 1, 1, 1, 1},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {2, 1, 2, 1, 1, 1, 1, 1},
 {2, 1, 2, 1, 1, 1, 1, 1},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {2, 1, 2, 1, 0, 0, 0, 0},
 {2, 1, 2, 1, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
 {2, 1, 2, 1, 0, 0, 0, 0},
 {2, 1, 2, 1, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0},
};

// #ifdef __CUDA_ARCH__
// __constant__ const uint8_t edge_manhattan_dist[12][24] = {
//   {0, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2},
//   {1, 2, 0, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 2, 2, 2},
//   {1, 3, 1, 2, 0, 3, 1, 2, 2, 3, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1},
//   {1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1, 2},
//   {1, 3, 2, 2, 2, 3, 2, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2},
//   {2, 2, 2, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 2, 2, 2, 1, 2},
//   {2, 3, 2, 2, 1, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1},
//   {2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 2, 1, 2, 2, 2},
//   {2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 3, 1, 3, 2, 3, 1, 3},
//   {2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 3, 0, 3, 1, 3, 2, 3},
//   {2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1, 3, 0, 3, 1, 3},
//   {2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 3, 0, 3},
// };
// #endif

// const uint8_t edge_manhattan_dist_cpu[12][24] = {
// {0, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2},
// {1, 2, 0, 3, 1, 2, 1, 3, 2, 2, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 2, 2, 2},
// {1, 3, 1, 2, 0, 3, 1, 2, 2, 3, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1},
// {1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 1, 2},
// {1, 3, 2, 2, 2, 3, 2, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2},
// {2, 2, 2, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 1, 3, 1, 2, 2, 2, 2, 2, 1, 2},
// {2, 3, 2, 2, 1, 3, 2, 2, 1, 3, 1, 2, 0, 3, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1},
// {2, 2, 1, 3, 2, 2, 2, 3, 1, 2, 1, 3, 1, 2, 0, 3, 2, 2, 1, 2, 1, 2, 2, 2},
// {2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 3, 1, 3, 2, 3, 1, 3},
// {2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 3, 0, 3, 1, 3, 2, 3},
// {2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1, 3, 0, 3, 1, 3},
// {2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 3, 0, 3},
// };

// #ifdef __CUDA_ARCH__
__device__ void TurnCube(uint8_t *cube, uint8_t turn) {
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
// #endif

__host__ void TurnCubeCPU(uint8_t *cube, uint8_t turn) {
  const uint8_t *positions = positions_on_face_cpu[turn / 3];
  uint8_t tmp[8];
  for (uint8_t i = 0; i < 8; i++) {
    tmp[i] = cube[positions[i]];
  }
  // For each corner position that this turn will affect
  for (uint8_t i = 0; i < NUM_POSITIONS_PER_FACE / 2; i++) {
    uint8_t new_value = tmp[i] + turn_orientation_cpu[turn][i];
    if (tmp[i] / 3 != new_value / 3) {
      new_value -= 3;
    }
    cube[turn_position_cpu[turn][i]] = new_value;
  }
  // For each edge position that this turn will affect
  for (uint8_t i = NUM_POSITIONS_PER_FACE / 2; i < NUM_POSITIONS_PER_FACE; i++) {
    uint8_t new_value = tmp[i] + turn_orientation_cpu[turn][i];
    if (tmp[i] / 2 != new_value / 2) {
      new_value -= 2;
    }
    cube[turn_position_cpu[turn][i]] = new_value;
  }
}

