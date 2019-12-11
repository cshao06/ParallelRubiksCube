#pragma once

#include <cstdint>

#define CUBE_ARR_SIZE 12+8

#define NUM_POSITIONS_PER_FACE 8

#define CORNER_POS_TO_FACET(p) (p * 2)
#define CORNER_FACET_TO_POS(f) (f / 2)
#define EDGE_POS_TO_FACET(p) ((p - kNumCornerPositions) * 2 + kNumCornerFacets)
#define EDGE_FACET_TO_POS(f) ((f - kNumCornerFacets) / 2 + kNumCornerPositions)

enum Turns {
  kU_ = 0,
  kUp,
  kU2,
  kD_,
  kDp,
  kD2,
  kF_,
  kFp,
  kF2,
  kB_,
  kBp,
  kB2,
  kL_,
  kLp,
  kL2,
  kR_,
  kRp,
  kR2,
  kNumTurns,
};

enum Faces {
  kU = 0,
  kD,
  kF,
  kB,
  kL,
  kR,
  kNumFaces,
};

enum Positions {
  kURF = 0,
  kUFL,
  kULB,
  kUBR,
  kDLF,
  kDFR,
  kDRB,
  kDBL,
  kNumCornerPositions = 8,
  kUF = 8,
  kUL,
  kUB,
  kUR,
  kDF,
  kDR,
  kDB,
  kDL,
  kFR,
  kFL,
  kBL,
  kBR,
  kNumPositions,
  kNumEdgePositions = kNumPositions - kNumCornerPositions,
};

enum Facets {
  kUrf = 0,
  kuRf,
  kurF,
  kUfl,
  kuFl,
  kufL,
  kUlb,
  kuLb,
  kulB,
  kUbr,
  kuBr,
  kubR,
  kDlf,
  kdLf,
  kdlF,
  kDfr,
  kdFr,
  kdfR,
  kDrb,
  kdRb,
  kdrB,
  kDbl,
  kdBl,
  kdbL,
  kNumCornerFacets = 24,
  kUf = 24,
  kuF,
  kUl,
  kuL,
  kUb,
  kuB,
  kUr,
  kuR,
  kDf,
  kdF,
  kDr,
  kdR,
  kDb,
  kdB,
  kDl,
  kdL,
  kFr,
  kfR,
  kFl,
  kfL,
  kBl,
  kbL,
  kBr,
  kbR,
  kNumFacets,
  kNumEdgeFacets = kNumFacets - kNumCornerFacets,
};

// #ifdef __CUDA_ARCH__
extern const char *turns_str[kNumTurns];
extern __constant__ const uint8_t positions_on_face[kNumFaces][NUM_POSITIONS_PER_FACE];
extern __constant__ const uint8_t turn_position[kNumTurns][NUM_POSITIONS_PER_FACE];
extern __constant__ const uint8_t turn_orientation[kNumTurns][NUM_POSITIONS_PER_FACE];
// #endif


extern const uint8_t positions_on_face_cpu[kNumFaces][NUM_POSITIONS_PER_FACE];
extern const uint8_t turn_position_cpu[kNumTurns][NUM_POSITIONS_PER_FACE];
extern const uint8_t turn_orientation_cpu[kNumTurns][NUM_POSITIONS_PER_FACE];

// #ifdef __CUDA_ARCH__
__device__ void TurnCube(uint8_t *cube, uint8_t turn);
// #endif

__host__ void TurnCubeCPU(uint8_t *cube, uint8_t turn);

