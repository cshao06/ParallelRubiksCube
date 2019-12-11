#pragma once

#include <cstdint>
#include <bitset>

#include "cube_notions.h"

using std::bitset;

/**
 * This class provides a map of all the possible scrambles of the 8 corner
 * cubies to the number of moves required to get the solved state.
 */
// typedef array<uint8_t, 8> perm_t;
typedef RubiksCube::FACE F;

array<uint8_t, 256> onesCountLookup;

CornerPatternDatabase();
uint32_t getDatabaseIndex(const RubiksCube& cube) const;
