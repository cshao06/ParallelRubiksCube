#include <iostream>
#include "cube_notions.h"
using namespace std;
// const uint8_t manhattan_dist[12][24] = {
//   [kUF] = {0, 3, 1, 2, 1, 3, 1, 2, 1, },
//   [kUL] = {},
//   [kUB] = {},
//   [kUR] = {},
//   [kDF] = {},
//   [kDR] = {},
//   [kDB] = {},
//   [kDL] = {},
//   [kFR] = {},
//   [kFL] = {},
//   [kBL] = {},
//   [kBR] = {},
// };


uint8_t manhattan_dist[12][24];

bool is_same_face(uint8_t pos1, uint8_t pos2, uint8_t *face_idx) {
  *face_idx = kNumFaces;
  for (int i = 0; i < kNumFaces; i++) {
    bool found1 = false;
    bool found2 = false;
    for (int j = 0; j < NUM_POSITIONS_PER_FACE; j++) {
      if (positions_on_face[i][j] == pos1) {
        found1 = true;
      }
      if (positions_on_face[i][j] == pos2) {
        found2 = true;
      }
    }
    if (found1 && found2) {
      *face_idx = i;
      return true;
    }
  }
  return false;
}

bool is_same_orientation(uint8_t facet1, uint8_t facet2, uint8_t face) {
  switch (face) {
    case kU:
    case kD:
    case kL:
    case kR:
      return facet1 % 2 == facet2 % 2;
      break;
    case kF:
    case kB:
      if ((facet1 >= kFr) == (facet2 >= kFr)) {
        return facet1 % 2 == facet2 % 2;
      } else {
        return facet1 % 2 != facet2 % 2;
      }
      break;
  }
  return false;
}

// Including the same position
bool is_opposite(uint8_t pos1, uint8_t pos2, uint8_t face) {
  if (pos1 == pos2) {
    return true;
  }
  if (face == kU || face == kD) {
    return (pos1 - pos2 + 4) % 2 == 0;
  } else {
    return (pos1 >= kFR) == (pos2 >= kFR);
  }
}

const uint8_t diagonal[12] = {kDB, kDR, kDF, kDL, kUB, kUL, kUF, kUR, kBL, kBR, kFR, kFL};

bool is_diagonal(uint8_t pos1, uint8_t pos2) {
  return pos2 == diagonal[pos1 - kNumCornerPositions];
}

void print_heuristic(uint8_t h[][24]) {
  cout << "Heuristic" << endl; 
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 24; j++) {
      cout << +h[i][j] << ", ";
    }
    cout << endl;
  }
}

int main(int argc, char *argv[]) {
  for (int ii = 0; ii < 12; ii++) {
    for (int jj = 0; jj < 24; jj++) {
      int i = ii + kNumCornerPositions;
      int j = jj + kNumCornerFacets;
      // cout << i << " " << j << endl;
      if (EDGE_POS_TO_FACET(i) == j) {
        manhattan_dist[ii][jj] = 0;
        continue;
      }
      uint8_t same_face;
      if (is_same_face(i, EDGE_FACET_TO_POS(j), &same_face)) {
        if (is_same_orientation(EDGE_POS_TO_FACET(i), j, same_face)) {
          manhattan_dist[ii][jj] = 1;
        } else {
          if (is_opposite(i, EDGE_FACET_TO_POS(j), same_face)) {
            manhattan_dist[ii][jj] = 3;
          } else {
            manhattan_dist[ii][jj] = 2;
          }
        }
      } else {
        if (is_diagonal(i, EDGE_FACET_TO_POS(j))) {
          if (j % 2 == 0) {
            manhattan_dist[ii][jj] = 2;
          } else {
            manhattan_dist[ii][jj] = 3;
          }
        } else {
          manhattan_dist[ii][jj] = 2;
        }
      }
    }
  }

  print_heuristic(manhattan_dist);
  return 0;
}
