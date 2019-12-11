#include <iostream>
#include <queue>
#include <vector>

#include "cube_notions.h"

using namespace std;

#define NUM_IDX (88179840)
#define NUM_STATES(i) (24-i*3)


// The original state
const uint8_t solved_cube[CUBE_ARR_SIZE] = {kUrf, kUfl, kUlb, kUbr, kDlf, kDfr, kDrb, kDbl, kUf, kUl, kUb, kUr, kDf, kDr, kDb, kDl, kFr, kFl, kBl, kBr};

// int main(int argc, char *argv[]) {
//   for (int idx = 0; idx < NUM_IDX; idx++) {
//   // for (int idx = 0; idx < 5; idx++) {
//     uint8_t cube[CUBE_ARR_SIZE];
//   }
// }

int main(int argc, char *argv[]) {
  for (int idx = 0; idx < NUM_IDX; idx++) {
  // for (int idx = 0; idx < 5; idx++) {
    uint8_t cube[CUBE_ARR_SIZE];
    for (int i = 0; i < 7; i++) {
      uint32_t moder = 1;
      uint32_t divider = 1;
      for (int j = 0; j < i; j++) {
        moder *= NUM_STATES(j);
      }
      for (int j = i + 1; j < 7; j++) {
        divider *= NUM_STATES(j);
      }
      cout << moder << " " << divider << endl;
      cube[i] = (idx % moder) / divider;
      bool taken[24] = {0};
      for (int k = 0; k < i; k++) {
        taken[cube[k] - (cube[k] % 3)] = true;
        taken[cube[k] - (cube[k] % 3) + 1] = true;
        taken[cube[k] - (cube[k] % 3) + 2] = true;
      }
      uint8_t count = cube[i];
      cout << +count << endl;
      for (int k = 0; k < 24; k++) {
        if (!taken[k]) {
          count--;
          if (count == 0) {
            cube[i] = k;
          }
        } 
      }
    }
    PrintCube(cube, 20);
  }
  return 0;
}

// void bfs() {
// 	// create a queue used to do BFS
// 	queue<int> q;
//   for (int i = 0; i < kNumTurns; i++) {
//   }
// }

// // Data structure to store graph edges
// struct Edge {
// 	int src, dest;
// };

// // Class to represent a graph object
// class Graph
// {
// public:
// 	// construct a vector of vectors to represent an adjacency list
// 	vector<vector<int>> adjList;

// 	// Graph Constructor
// 	Graph(vector<Edge> const &edges, int N)
// 	{
// 		// resize the vector to N elements of type vector<int>
// 		adjList.resize(N);

// 		// add edges to the undirected graph
// 		for (auto &edge: edges)
// 		{
// 			adjList[edge.src].push_back(edge.dest);
// 			adjList[edge.dest].push_back(edge.src);
// 		}
// 	}
// };

// // Perform BFS on graph starting from vertex v
// void BFS(Graph const &graph, int v, vector<bool> &discovered)
// {
// 	// create a queue used to do BFS
// 	queue<int> q;

// 	// mark source vertex as discovered
// 	discovered[v] = true;

// 	// push source vertex into the queue
// 	q.push(v);

// 	// run till queue is not empty
// 	while (!q.empty())
// 	{
// 		// pop front node from queue and print it
// 		v = q.front();
// 		q.pop();
// 		cout << v << " ";

// 		// do for every edge (v -> u)
// 		for (int u : graph.adjList[v])
// 			if (!discovered[u])
// 			{
// 				// mark it discovered and push it into queue
// 				discovered[u] = true;
// 				q.push(u);
// 			}
// 	}
// }

// // Iterative C++ implementation of Breadth first search
// int main()
// {
// 	// vector of graph edges as per above diagram
// 	vector<Edge> edges = {
// 		{1, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {5, 9},
// 		{5, 10}, {4, 7}, {4, 8}, {7, 11}, {7, 12}
// 		// vertex 0, 13 and 14 are single nodes
// 	};

// 	// Number of nodes in the graph
// 	int N = 15;

// 	// create a graph from edges
// 	Graph graph(edges, N);

// 	// stores vertex is discovered or not
// 	vector<bool> discovered(N, false);

// 	// Do BFS traversal from all undiscovered nodes to
// 	// cover all unconnected components of graph
// 	for (int i = 0; i < N; i++) {
// 		if (discovered[i] == false) {
// 			// start BFS traversal from vertex i
// 			BFS(graph, i, discovered);
// 		}
// 	}

// 	return 0;
// }
