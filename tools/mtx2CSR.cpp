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
    std::vector<std::tuple<int, int, int>>& changedEdges) {

    std::vector<std::tuple<int, int, int>> newGraph = originalGraph;
    std::set<std::pair<int, int>> existingEdges;
    
    for (const auto& [src, dest, weight] : originalGraph) {
        existingEdges.insert({src, dest});
    }

    std::srand(std::time(nullptr));

    for (int i = 0; i < numChanges; ++i) {
        int action = std::rand() % 2;

        if (action == 0 && !newGraph.empty()) {
            int index = std::rand() % newGraph.size();
            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);
            std::get<2>(newGraph[index]) = newWeight;
            changedEdges.push_back(newGraph[index]);
        } else {
            int src, dest;
            do {
                src = 1 + std::rand() % numVertices;
                dest = 1 + std::rand() % numVertices;
            } while (src == dest || existingEdges.find({src, dest}) != existingEdges.end());

            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);

            newGraph.emplace_back(src, dest, newWeight);
            changedEdges.emplace_back(src, dest, newWeight);
            existingEdges.insert({src, dest});
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

void dijkstra(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers,
              int source, std::vector<int>& parent) {
    std::vector<int> dist(row_pointers.size() - 1, INF);
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    
    dist[source] = 0;
    pq.push({0, source});
    parent[source] = source; // source is the parent of itself

    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();

        if (d > dist[u]) continue;

        int start = row_pointers[u];
        int end = row_pointers[u + 1];
        for (int i = start; i < end; ++i) {
            int v = column_indices[i];
            int alt = dist[u] + values[i]; // assuming all edges have weight 1
            if (alt < dist[v]) {
                dist[v] = alt;
                pq.push({dist[v], v});
                parent[v] = u; // set parent
            }
        }
    }
}

int main() {

//Input Graph to CSR

    sortAndSaveMTX("graph.mtx", "sorted_graph.mtx");

    std::ifstream file("sorted_graph.mtx");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::string line;
    int numRows, numCols, numNonZero;
    
    // Skip comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    std::vector<int> values(numNonZero);
    std::vector<int> column_indices(numNonZero);
    std::vector<int> row_pointers(numRows + 1, 0);

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
    // Print CSR representation of SPT
    std::cout << "CSR representation of the Predecessor:\n";
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

// Predecessor
    std::ifstream file2("sorted_graph.mtx");

    if (!file2.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::string line2;

    
    // Skip comments
    do {
        std::getline(file2, line2);
    } while (line2[0] == '%');

    std::stringstream ss2(line2);
    ss2 >> numRows >> numCols >> numNonZero;

    std::vector<int> values_pred(numNonZero);
    std::vector<int> column_indices_pred(numNonZero);
    std::vector<int> row_pointers_pred(numRows + 1, 0);

    nnz = 0;
    current_row = 0;
    
    for (int i = 0; i < numNonZero; ++i) {
        file2 >> row >> col >> val;

        // Convert to 0-based indexing
        row -= 1;
        col -= 1;

        while (row > current_row) {
            row_pointers_pred[current_row + 1] = nnz;
            current_row++;
        }

        values_pred[nnz] = val;
        column_indices_pred[nnz] = col;
        nnz++;
    }

    // Add the last row_pointer
    row_pointers_pred[current_row + 1] = nnz;

    // Close the file
    file2.close();
    // Print CSR representation of SPT
    std::cout << "CSR representation of the Predecessor:\n";
    std::cout << "Values: ";
    for (int val : values_pred) {
        std::cout << val << " ";
    }
    std::cout << "\nColumn Indices: ";
    for (int col : column_indices_pred) {
        std::cout << col << " ";
    }
    std::cout << "\nRow Pointers: ";
    for (int row_ptr : row_pointers_pred) {
        std::cout << row_ptr << " ";
    }
    std::cout << std::endl;

//Find SSSP tree

    // Test the function
    for (int vertex = 0; vertex < numRows; ++vertex) {
        find_neighbors(row_pointers, column_indices, vertex);
    }

    std::vector<int> parent(row_pointers.size() - 1, -1); // Initialize parent array

    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values, column_indices, row_pointers, 0, parent);

    // Generate CSR representation for the Shortest Path Tree (SPT)
    std::vector<int> spt_values, spt_column_indices, spt_row_pointers;
    spt_row_pointers.push_back(0);
    nnz = 0;

    for (int u = 0; u < parent.size(); ++u) {
        if (parent[u] != -1) {
            spt_values.push_back(1); // assuming all edges have weight 1
            spt_column_indices.push_back(u);
            nnz++;
        }
        spt_row_pointers.push_back(nnz);
    }

    // Print CSR representation of SPT
    std::cout << "CSR representation of the Shortest Path Tree:\n";
    std::cout << "Values: ";
    for (int val : spt_values) {
        std::cout << val << " ";
    }
    std::cout << "\nColumn Indices: ";
    for (int col : spt_column_indices) {
        std::cout << col << " ";
    }
    std::cout << "\nRow Pointers: ";
    for (int row_ptr : spt_row_pointers) {
        std::cout << row_ptr << " ";
    }
    std::cout << std::endl;

// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    if (originalGraph.empty()) {
        return 1;
    }

    int numVertices = 4;  // Should be determined from the MTX file or another source
    int numChanges = 3;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges);

    writeMTX("new_graph.mtx", newGraph, numVertices);
    writeMTX("changed_edges.mtx", changedEdges, numVertices);

    return 0;
}
