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
    // std::cout << "Neighbors of vertex " << vertex << ": ";
    // for (int i = start; i < end; ++i) {
    //     std::cout << column_indices[i] << " ";
    // }
    // std::cout << std::endl;
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
  

    std::vector<int> changed_values;
    std::vector<int> changed_column_indices;
    std::vector<int> changed_row_pointers;

    changed_row_pointers.push_back(0);
    int nnz = 0;

    // Assuming vertices are 0-indexed
    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        for (const auto& [src, dest, weight] : changedEdges) {
            if (src - 1 == u) {
                changed_values.push_back(weight);
                changed_column_indices.push_back(dest - 1);
                nnz++;
            }
        }
        changed_row_pointers.push_back(nnz);
    }

    // Original logic of updateShortestPath adapted to use changed_* vectors
    std::queue<int> affectedNodes;
    std::vector<bool> isAffected(row_pointers.size() - 1, false);

    for (int u = 0; u < row_pointers.size() - 1; ++u) {
        int start = changed_row_pointers[u];
        int end = changed_row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = changed_column_indices[i];
            int alt = dist[u] + changed_values[i];
            if (alt < dist[v]) {
                dist[v] = alt;
                parent[v] = u;
                isAffected[v] = true;
                affectedNodes.push(v);
            }
        }
    }

    // Propagate changes
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

// #include <CL/sycl.hpp>

// void updateShortestPath(const std::vector<int>& values, const std::vector<int>& column_indices, 
//                         const std::vector<int>& row_pointers, std::vector<int>& dist, 
//                         std::vector<int>& parent, const std::vector<std::tuple<int, int, int>>& changedEdges) 
// {
//     // SYCL queue for running kernels
//     cl::sycl::queue q;

//     // Buffers for storing data
//     cl::sycl::buffer<int> buf_values(values.data(), cl::sycl::range<1>(values.size()));
//     cl::sycl::buffer<int> buf_column_indices(column_indices.data(), cl::sycl::range<1>(column_indices.size()));
//     cl::sycl::buffer<int> buf_row_pointers(row_pointers.data(), cl::sycl::range<1>(row_pointers.size()));
//     cl::sycl::buffer<int> buf_dist(dist.data(), cl::sycl::range<1>(dist.size()));
//     cl::sycl::buffer<int> buf_parent(parent.data(), cl::sycl::range<1>(parent.size()));

//     // Affected Nodes and Flags
//     std::queue<int> affectedNodes;
//     std::vector<bool> isAffected(row_pointers.size() - 1, false);

//     // Handle changed edges (this part remains serial)
//     for (const auto& [src, dest, weight] : changedEdges) {
//         int alt = dist[src] + weight;
//         if (alt < dist[dest]) {
//             dist[dest] = alt;
//             parent[dest] = src;
//             isAffected[dest] = true;
//             affectedNodes.push(dest);
//         }
//     }

//     // SYCL Code to propagate changes (very simplified)
//     while (!affectedNodes.empty()) {
//         int u = affectedNodes.front();
//         affectedNodes.pop();
//         isAffected[u] = false;

//         int start = row_pointers[u];
//         int end = row_pointers[u + 1];

//         // Submit kernel to queue
//         q.submit([&](cl::sycl::handler& cgh) {
//             auto acc_values = buf_values.get_access<cl::sycl::access::mode::read>(cgh);
//             auto acc_column_indices = buf_column_indices.get_access<cl::sycl::access::mode::read>(cgh);
//             auto acc_dist = buf_dist.get_access<cl::sycl::access::mode::read_write>(cgh);
//             auto acc_parent = buf_parent.get_access<cl::sycl::access::mode::write>(cgh);

//             cgh.parallel_for(cl::sycl::range<1>(end - start), [=](cl::sycl::id<1> i) {
//                 int idx = i[0] + start;
//                 int v = acc_column_indices[idx];
//                 int alt = acc_dist[u] + acc_values[idx];
//                 if (alt < acc_dist[v]) {
//                     acc_dist[v] = alt;
//                     acc_parent[v] = u;
//                     // Note: Handling of affected nodes is left out as it's more complex in parallel setting
//                 }
//             });
//         });
//     }

//     // Copy data back to host if needed (depends on the specifics of your SYCL implementation and hardware)
//     // e.g., you might use host accessors, or update the original vectors using buffer::get_access()
// }


#include <vector>
#include <queue>
#include <utility>
#include <limits>

void dijkstra2(const std::vector<int>& values, const std::vector<int>& column_indices, 
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
    // // Print CSR representation of SPT
    // std::cout << "CSR representation of the Graph:\n";
    // std::cout << "Values: ";
    // for (int val : values) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\nColumn Indices: ";
    // for (int col : column_indices) {
    //     std::cout << col << " ";
    // }
    // std::cout << "\nRow Pointers: ";
    // for (int row_ptr : row_pointers) {
    //     std::cout << row_ptr << " ";
    // }
    // std::cout << std::endl;

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
    // std::cout << "CSR representation of the Predecessor:\n";
    // std::cout << "Values: ";
    // for (int val : values_pred) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\nColumn Indices: ";
    // for (int col : column_indices_pred) {
    //     std::cout << col << " ";
    // }
    // std::cout << "\nRow Pointers: ";
    // for (int row_ptr : row_pointers_pred) {
    //     std::cout << row_ptr << " ";
    // }
    // std::cout << std::endl;
    

//Find SSSP tree and store in mtx file

    // Test the function
    for (int vertex = 0; vertex < numRows; ++vertex) {
        find_neighbors(row_pointers, column_indices, vertex);
    }

    std::vector<int> parent(row_pointers.size() - 1, -1); // Initialize parent array
    std::vector<int> dist(row_pointers.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra2(values, column_indices, row_pointers, 0, dist, parent);


    std::ofstream outfile("SSSP_Tree.mtx");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
    }

    outfile << parent.size() << " " << parent.size() << " " << parent.size() - 1 << std::endl;

    // index starting from 1 as source is a parent of source case. 
    // The value assignment has bug
    for (int i = 1; i < parent.size(); i++) {

        int val = -1; 
        int start = row_pointers[parent[i]]; 
        int end = row_pointers[parent[i] + 1];
        for ( ; start < end ; start++ )
        {
            if ( column_indices[start] == i )
            {
                val = values[start];
            }
        }
        outfile << parent[i] + 1 << " " << i + 1 << " " << val << std::endl;
    }

// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    if (originalGraph.empty()) {
        return 1;
    }

    int numVertices = row_pointers.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges = 3;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges);

    writeMTX("new_graph.mtx", newGraph, numVertices);
    writeMTX("changed_edges.mtx", changedEdges, numVertices);

// Update shortest path code
    std::cout << "CSR representation: Before updateShortestPath:\n";
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
    std::cout<< "\n";
    // Now call the updateShortestPath function
    updateShortestPath(values, column_indices, row_pointers, dist, parent, changedEdges);

    std::vector<int> new_values, new_column_indices, new_row_pointers;

    new_row_pointers.push_back(0);
    nnz = 0;
    for (int u = 0; u < numRows; ++u) {
        if (parent[u] != -1) {
            new_values.push_back(dist[u]);  // storing the shortest distance as the value
            new_column_indices.push_back(parent[u]);  // storing the parent as the column index
            nnz++;
        }
        new_row_pointers.push_back(nnz);
    }

    // Print new CSR representation
    std::cout << "New CSR representation after applying changedEdges:\n";
    std::cout << "Values: ";
    for (int val : new_values) {
        std::cout << val << " ";
    }
    std::cout << "\nColumn Indices: ";
    for (int col : new_column_indices) {
        std::cout << col << " ";
    }
    std::cout << "\nRow Pointers: ";
    for (int row_ptr : new_row_pointers) {
        std::cout << row_ptr << " ";
    }
    std::cout << std::endl;

    return 0;
}
