#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <utility>

void find_neighbors(const std::vector<int>& row_pointers, const std::vector<int>& column_indices, int vertex) {
    int start = row_pointers[vertex];
    int end = row_pointers[vertex + 1];
    std::cout << "Neighbors of vertex " << vertex << ": ";
    for (int i = start; i < end; ++i) {
        std::cout << column_indices[i] << " ";
    }
    std::cout << std::endl;
}




const int INF = 1e9; // Representing infinity

std::vector<std::tuple<int, int, int>> readMTX(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<std::tuple<int, int, int>> graph;
    
    if (!infile.is_open()) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return graph;
    }
    
    std::string line;
    do {
        std::getline(infile, line);
    } while (line[0] == '%');
    
    int numRows, numCols, numNonZero;
    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;
    
    int row, col, weight;
    while (infile >> row >> col >> weight) {
        graph.emplace_back(row, col, weight);
    }
    
    return graph;
}

void writeMTX(const std::string& filename, const std::vector<std::tuple<int, int, int>>& graph, int numVertices) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file " << filename << std::endl;
        return;
    }
    
    outfile << numVertices << " " << numVertices << " " << graph.size() << "\n";
    
    for (const auto& [src, dest, weight] : graph) {
        outfile << src << " " << dest << " " << weight << "\n";
    }
}

std::vector<std::tuple<int, int, int>> generateChangedGraph(
    const std::vector<std::tuple<int, int, int>>& originalGraph,
    int numVertices,
    int numChanges,
    int minWeight,
    int maxWeight,
    std::vector<std::tuple<int, int, int>>& changedEdges,
    float deletionPercentage // e.g., 0.2 for 20%
) {
    std::vector<std::tuple<int, int, int>> newGraph = originalGraph;
    std::set<std::pair<int, int>> existingEdges;

    for (const auto& [src, dest, weight] : originalGraph) {
        existingEdges.insert({src, dest});
    }

    std::srand(std::time(nullptr));

    int numDeletions = static_cast<int>(numChanges * deletionPercentage);
    int numOtherActions = numChanges - numDeletions;

    for (int i = 0; i < numChanges; ++i) {
        int action;
        
        if (numDeletions > 0) {
            action = 2;
            numDeletions--;
        } else if (numOtherActions > 0) {
            action = std::rand() % 2; // 0 or 1
            numOtherActions--;
        }

        if (action == 0 && !newGraph.empty()) {
            // Change Weight
            int index = std::rand() % newGraph.size();
            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);
            std::get<2>(newGraph[index]) = newWeight;
            changedEdges.push_back(newGraph[index]);
        } else if (action == 1) {
            // Add Edge
            int src, dest;
            do {
                src = 1 + std::rand() % numVertices;
                dest = 1 + std::rand() % numVertices;
            } while (src == dest || existingEdges.find({src, dest}) != existingEdges.end());

            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);

            newGraph.emplace_back(src, dest, newWeight);
            changedEdges.emplace_back(src, dest, newWeight);
            existingEdges.insert({src, dest});
        } else {
            // Delete Edge (by setting the weight to the negative of the current weight)
            if (!newGraph.empty()) {
                int index = std::rand() % newGraph.size();
                int curWeight = std::get<2>(newGraph[index]);
                std::get<2>(newGraph[index]) = -curWeight;
                changedEdges.push_back(newGraph[index]);
            }
        }
    }

    std::sort(newGraph.begin(), newGraph.end());
    std::sort(changedEdges.begin(), changedEdges.end());

    return newGraph;
}

void sortAndSaveMTX(const std::string& input_filename, const std::string& output_filename) {
    std::ifstream infile(input_filename);

    if (!infile.is_open()) {
        std::cerr << "Failed to open the input file." << std::endl;
        return;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    // Skip comments
    do {
        std::getline(infile, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    std::vector<std::tuple<int, int, int>> edges;  // source, destination, weight
    int row, col, weight;
    int prev_row = -1;
    bool is_sorted = true;

    while (infile >> row >> col >> weight) {
        if (row < prev_row) {
            is_sorted = false;
        }
        prev_row = row;
        edges.emplace_back(row, col, weight);
    }

    infile.close();

    if (!is_sorted) {
        std::sort(edges.begin(), edges.end());
    }

    // Save to a new MTX file
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << numRows << " " << numCols << " " << numNonZero << std::endl;

    for (const auto& [r, c, w] : edges) {
        outfile << r << " " << c << " " << w << std::endl;
    }

    outfile.close();
}

// void dijkstra(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers,
//               int source, std::vector<int>& parent) {
//     std::vector<int> dist(row_pointers.size() - 1, INF);
//     std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    
//     dist[source] = 0;
//     pq.push({0, source});
//     parent[source] = source; // source is the parent of itself

//     while (!pq.empty()) {
//         int u = pq.top().second;
//         int d = pq.top().first;
//         pq.pop();

//         if (d > dist[u]) continue;

//         int start = row_pointers[u];
//         int end = row_pointers[u + 1];
//         for (int i = start; i < end; ++i) {
//             int v = column_indices[i];
//             int alt = dist[u] + values[i]; // assuming all edges have weight 1
//             if (alt < dist[v]) {
//                 dist[v] = alt;
//                 pq.push({dist[v], v});
//                 parent[v] = u; // set parent
//             }
//         }
//     }
// }
void updateShortestPath(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers,
                        std::vector<int>& dist, std::vector<int>& parent,
                        const std::vector<std::tuple<int, int, int>>& changedEdges) {

    std::cout << "Distance before" <<std::endl;
    for (int i = 0; i < dist.size(); i++) {
        std::cout <<dist[i]<< " ";
    }
    std::cout<<std::endl;

    std::cout << "Parent before " <<std::endl;
    for (int i = 0; i < parent.size(); i++) {
        std::cout <<parent[i]<< " ";
    }
    std::cout<<std::endl;
  
    // Convert changedEdges to CSR for insertion and deletion separately
    std::vector<int> insert_values, delete_values;
    std::vector<int> insert_column_indices, delete_column_indices;
    std::vector<int> insert_row_pointers, delete_row_pointers;
    insert_row_pointers.push_back(0);
    delete_row_pointers.push_back(0);

    int insert_nnz = 0, delete_nnz = 0;

    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        for (const auto& [src, dest, weight] : changedEdges) {
            if (src - 1 == u) {
                if (weight >= 0) {
                    insert_values.push_back(weight);
                    insert_column_indices.push_back(dest - 1);
                    insert_nnz++;
                } else {
                    delete_values.push_back(-weight);
                    delete_column_indices.push_back(dest - 1);
                    delete_nnz++;
                }
            }
        }
        insert_row_pointers.push_back(insert_nnz);
        delete_row_pointers.push_back(delete_nnz);
    }


    insert_row_pointers.push_back(0);
    int nnz = 0;

    // Assuming vertices are 1-indexed
    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        for (const auto& [src, dest, weight] : changedEdges) {
            if (src - 1 == u) {
                insert_values.push_back(weight);
                insert_column_indices.push_back(dest - 1);
                nnz++;
            }
        }
        insert_row_pointers.push_back(nnz);
    }

    // Original logic of updateShortestPath adapted to use changed_* vectors
    std::queue<int> affectedNodes;
    std::vector<bool> isAffected(row_pointers.size() - 1, false);

    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        int start = insert_row_pointers[u];
        int end = insert_row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = insert_column_indices[i];
            int alt = dist[u] + insert_values[i];
            if (alt < dist[v]) {
                dist[v] = alt;
                parent[v] = u;
                isAffected[v] = true;
                affectedNodes.push(v);
            }
        }
    }

    // Propagate changes for insertion
    while (!affectedNodes.empty()) {
        int u = affectedNodes.front();
        affectedNodes.pop();
        isAffected[u] = false;

        int start = row_pointers[u];
        int end = row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = column_indices[i];
            int alt = dist[u] + values[i];
            if (alt < dist[v]) {
                dist[v] = alt;
                parent[v] = u;
                if (!isAffected[v]) {
                    isAffected[v] = true;
                    affectedNodes.push(v);
                }
            }
        }
    }

    // Handle deletions
    std::queue<int> affectedNodesForDeletion;
    std::vector<bool> isAffectedForDeletion(row_pointers.size() - 1, false);

    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        int start = delete_row_pointers[u];
        int end = delete_row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = delete_column_indices[i];
            if (parent[v] == u) { // if this deleted edge was part of the shortest path
                dist[v] = INT_MAX; // Invalidate the shortest distance
                parent[v] = -1;   // Invalidate the parent
                isAffectedForDeletion[v] = true;
                affectedNodesForDeletion.push(v);
            }
        }
    }

    // Recompute shortest paths for affected nodes
    while (!affectedNodesForDeletion.empty()) {
        int u = affectedNodesForDeletion.front();
        affectedNodesForDeletion.pop();
        isAffectedForDeletion[u] = false;

        int start = row_pointers[u];
        int end = row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = column_indices[i];
            int alt = (dist[u] == INT_MAX) ? INT_MAX : dist[u] + values[i];
            if (alt < dist[v]) {
                dist[v] = alt;
                parent[v] = u;
                if (!isAffectedForDeletion[v]) {
                    isAffectedForDeletion[v] = true;
                    affectedNodesForDeletion.push(v);
                }
            }
        }
    }

    std::cout << "Distance after" <<std::endl;
    for (int i = 0; i < dist.size(); i++) {
        std::cout <<dist[i]<< " ";
    }
    std::cout<<std::endl;

    std::cout << "Parent after " <<std::endl;
    for (int i = 0; i < parent.size(); i++) {
        std::cout <<parent[i]<< " ";
    }
    std::cout<<std::endl;

}




#include <vector>
#include <queue>
#include <utility>
#include <limits>

void dijkstra(const std::vector<int>& values, const std::vector<int>& column_indices, 
              const std::vector<int>& row_pointers, int src,
              std::vector<int>& dist, std::vector<int>& parent)
{
    // Initialize the distance vector
    int n = row_pointers.size() - 1;
    dist.resize(n, std::numeric_limits<int>::max());
    dist[src] = 0;

    // Initialize the parent vector
    parent.resize(n, -1);
    parent[src] = src;

    // Priority queue to store {distance, vertex} pairs
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        int uDist = pq.top().first;
        pq.pop();

        // Ignore if distance in the queue is outdated
        if (uDist > dist[u]) continue;

        int start = row_pointers[u];
        int end = row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = column_indices[i];
            int weight = values[i];
            
            // Relaxation step
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}
int numRows, numCols, numNonZero;
bool readMTXToCSR(const std::string& filename, std::vector<int>& values, std::vector<int>& column_indices, std::vector<int>& row_pointers) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return false;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    // Skip comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    values.resize(numNonZero);
    column_indices.resize(numNonZero);
    row_pointers.resize(numRows + 1, 0);

    int row, col, val;
    int nnz = 0;
    int current_row = 0;

    for (int i = 0; i < numNonZero; ++i) {
        file >> row >> col >> val;

        // Convert to 0-based indexing
        row -= 1;
        col -= 1;

        while (row > current_row) {
            row_pointers[current_row + 1] = nnz;
            current_row++;
        }

        values[nnz] = val;
        column_indices[nnz] = col;
        nnz++;
    }

    // Add the last row_pointer
    row_pointers[current_row + 1] = nnz;

    // Close the file
    file.close();

    return true;
}
// Function to print CSR representation
void printCSRRepresentation(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers) {
    std::cout << "CSR representation of the Graph:\n";
    std::cout << "Values: ";
    for (int val : values) {
        std::cout << val << " ";
    }
    std::cout << "\nColumn Indices: ";
    for (int col : column_indices) {
        std::cout << col << " ";
    }
    std::cout << "\nRow Pointers: ";
    for (int row_ptr : row_pointers) {
        std::cout << row_ptr << " ";
    }
    std::cout << std::endl;
}

void saveSSSPTreeToFile(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers, const std::vector<int>& parent) {
    std::ofstream outfile("SSSP_Tree.mtx");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << parent.size() << " " << parent.size() << " " << parent.size() - 1 << std::endl;

    // index starting from 1 as the source is a parent of the source case
    for (int i = 1; i < parent.size(); i++) {
        int val = -1;
        int start = row_pointers[parent[i]];
        int end = row_pointers[parent[i] + 1];
        for (; start < end; start++) {
            if (column_indices[start] == i) {
                val = values[start];
            }
        }
        outfile << parent[i] + 1 << " " << i + 1 << " " << val << std::endl;
    }

    outfile.close();
}
void saveSSSPTreeToFile(const std::string& fileName, const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers, const std::vector<int>& parent) {
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << parent.size() << " " << parent.size() << " " << parent.size() - 1 << std::endl;

    // index starting from 1 as the source is a parent of the source case
    for (int i = 1; i < parent.size(); i++) {
        int val = -1;
        int start = row_pointers[parent[i]];
        int end = row_pointers[parent[i] + 1];
        for (; start < end; start++) {
            if (column_indices[start] == i) {
                val = values[start];
            }
        }
        outfile << parent[i] + 1 << " " << i + 1 << " " << val << std::endl;
    }

    outfile.close();
}

int main() {

//Input Graph to CSR

    sortAndSaveMTX("graph.mtx", "sorted_graph.mtx");

    std::vector<int> values;
    std::vector<int> column_indices;
    std::vector<int> row_pointers;

    readMTXToCSR("sorted_graph.mtx", values, column_indices, row_pointers);
    printCSRRepresentation(values, column_indices, row_pointers);


//Find SSSP tree and store in mtx file


    std::vector<int> parent(row_pointers.size() - 1, -1); // takes child and returns it's parent
    std::vector<int> dist(row_pointers.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values, column_indices, row_pointers, 0, dist, parent);

    saveSSSPTreeToFile("SSSP_Tree.mtx", values, column_indices, row_pointers, parent);

    

// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    int numVertices = row_pointers.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges = 3;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    float deletePercentage = 0.0f;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges, deletePercentage);
    // writeMTX by default sort by row for easy reading
    writeMTX("new_graph.mtx", newGraph, numVertices); 
    writeMTX("changed_edges.mtx", changedEdges, numVertices);

// Update shortest path code
    std::cout << "CSR representation: Before updateShortestPath:\n";
    printCSRRepresentation(values, column_indices, row_pointers);

    // Now call the updateShortestPath function
    updateShortestPath(values, column_indices, row_pointers, dist, parent, changedEdges);

    std::vector<int> new_values, new_column_indices, new_row_pointers;

    new_row_pointers.push_back(0);
    int nnz = 0;
    for (int u = 0; u < numRows; ++u) {
        if (parent[u] != -1) {
            new_values.push_back(dist[u]);  // storing the shortest distance as the value
            new_column_indices.push_back(parent[u]);  // storing the parent as the column index
            nnz++;
        }
        new_row_pointers.push_back(nnz);
    }

    // Print new CSR representation
    printCSRRepresentation(new_values, new_column_indices, new_row_pointers);

    return 0;
}

//clang++ -std=c++17 mtx2CSR.cpp  && ./a.out