#include "corner_heuristic.h"

/**
   * Given a cube, get an index into the pattern database.
   */
uint32_t getDatabaseIndex(const uint8_t *cube) const {
  // typedef RubiksCubeIndexModel::CORNER CORNER;

  // const RubiksCubeIndexModel& iCube = static_cast<const RubiksCubeIndexModel&>(cube);

  // The permutation of the 8 corners.
  perm_t cornerPerm =
  {
    iCube.getCornerIndex(CORNER::ULB),
    iCube.getCornerIndex(CORNER::URB),
    iCube.getCornerIndex(CORNER::URF),
    iCube.getCornerIndex(CORNER::ULF),
    iCube.getCornerIndex(CORNER::DLF),
    iCube.getCornerIndex(CORNER::DLB),
    iCube.getCornerIndex(CORNER::DRB),
    iCube.getCornerIndex(CORNER::DRF)
  };

  // Compute the Lehmer code using Korf's linear algorithm.  It's discussed
  // in his paper, Large-Scale Parallel Breadth-First Search
  // (https://www.aaai.org/Papers/AAAI/2005/AAAI05-219.pdf).
  //
  // "We scan the permutation from left to right, constructing a bit string
  // of length n, indicating which elements of the permutation we've seen
  // so far. Initially the string is all zeros.  As each element of the
  // permutation is encountered, we use it as an index into the bit string
  // and set the corresponding bit to one. When we encounter element k in
  // the permutation, to determine the number of elements less than k to
  // its left, we need to know the number of ones in the first k bits of
  // our bit string. We extract the first k bits by right shifting the
  // string by n − k. This reduces the problem to: given a bit string,
  // count the number of one bits in it.
  // We solve this problem in constant time by using the bit string as an
  // index into a precomputed table, containing the number of ones in the
  // binary representation of each index."
  uint8_t lehmer[8];
  bitset<8> seen;

  lehmer[0] = cornerPerm[0];
  seen[7 - cornerPerm[0]] = 1;
  lehmer[7] = 0;

  for (unsigned i = 1; i < 7; ++i)
  {
    // std::bitset indexes right-to-left.
    seen[7 - cornerPerm[i]] = 1;

    uint8_t numOnes = this->onesCountLookup[seen.to_ulong() >> (8 - cornerPerm[i])];

    lehmer[i] = cornerPerm[i] - numOnes;
  }

  // Now convert the Lehmer code to a base-10 number.  To do so,
  // multiply each digit by it's corresponding factorial base.
  // E.g. the permutation 120 has a Lehmer code of 110, which is
  // 1 * 2! + 1 * 1! + 0 * 0! = 3.
  uint32_t index =
    lehmer[0] * 5040 +
    lehmer[1] * 720 +
    lehmer[2] * 120 +
    lehmer[3] * 24 +
    lehmer[4] * 6 +
    lehmer[5] * 2 +
    lehmer[6];

  // Now get the orientation of the corners.  7 corner orientations dictate
  // the orientation of the 8th, so only 7 need to be stored.
  uint8_t cornerOrientations[7] =
  {
    iCube.getCornerOrientation(CORNER::ULB),
    iCube.getCornerOrientation(CORNER::URB),
    iCube.getCornerOrientation(CORNER::URF),
    iCube.getCornerOrientation(CORNER::ULF),
    iCube.getCornerOrientation(CORNER::DLF),
    iCube.getCornerOrientation(CORNER::DLB),
    iCube.getCornerOrientation(CORNER::DRB)
  };

  // Treat the orientations as a base-3 number, and convert it
  // to base-10.
  uint32_t orientationNum =
    cornerOrientations[0] * 729 +
    cornerOrientations[1] * 243 +
    cornerOrientations[2] * 81 +
    cornerOrientations[3] * 27 +
    cornerOrientations[4] * 9 +
    cornerOrientations[5] * 3 +
    cornerOrientations[6];

  // Combine the permutation and orientation into a single index.
  // p * 3^7 + o;
  return index * 2187 + orientationNum;
}
