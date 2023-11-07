#include <bits/stdc++.h>
#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace std;
const int INF = 1e9;
using Graph = std::vector<std::vector<std::pair<int, int>>>;

struct CSR {
    vector<int> values;
    vector<int> row_ptr;
    vector<int> col_idx;
};
struct Edge {
    int u;
    int v;
    float weight;
};

class Tree {
public:
    virtual const std::vector<int>& getParents() const = 0;
    virtual ~Tree() = default;
};

class Tree1 : public Tree {
    std::vector<int> parents;
public:
    Tree1(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree2 : public Tree {
    std::vector<int> parents;
public:
    Tree2(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree3 : public Tree {
    std::vector<int> parents;
public:
    Tree3(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};
// Function to add an edge to the graph
void addEdge(std::vector<Edge>& edges, int u, int v, double pref) {
    edges.push_back({u, v, static_cast<float>(pref)});
}

// Function to construct the graph
std::vector<Edge> constructGraph(const std::vector<Tree*>& trees, const std::vector<double>& Pref) {
    std::vector<Edge> edges;

    for (size_t index = 0; index < trees.size(); ++index) {
        const auto& tree = trees[index];
        const std::vector<int>& parents = tree->getParents();

        for (int i = 1; i < parents.size(); ++i) {
            if (parents[i] != 0) {
                addEdge(edges, parents[i], i, Pref[index]);
            }
        }
    }

    return edges;
}

// SYCL kernel for updating the weights of the edges
class UpdateWeightsKernel;

void updateWeights(std::vector<Edge>& edges, const std::vector<double>& Pref) {
    // Create a SYCL queue to manage commanded execution.
    cl::sycl::queue q(cl::sycl::gpu_selector_v);

    // Create buffers for edges and preferences.
    buffer<Edge, 1> edges_buf(edges.data(), range<1>(edges.size()));
    buffer<double, 1> pref_buf(Pref.data(), range<1>(Pref.size()));

    // Submit command group to queue to execute kernel.
    q.submit([&](handler& cgh) {
        // Create accessors to buffers.
        auto edges_acc = edges_buf.get_access<access::mode::read_write>(cgh);
        auto pref_acc = pref_buf.get_access<access::mode::read>(cgh);

        // Execute kernel.
        cgh.parallel_for<UpdateWeightsKernel>(range<1>(edges.size()), [=](id<1> idx) {
            edges_acc[idx].weight = static_cast<float>(-1/2.0);
        });
    });
    // Wait for the queue to finish.
    q.wait();
}
//Done
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
void writeMTX(const std::string& filename, const std::vector<std::tuple<int, int, int>>& graph, int numVertices, bool isGraph) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file " << filename << std::endl;
        return;
    }
    
    outfile << numVertices << " " << numVertices << " " << graph.size() << "\n";
    
    for (const auto& [src, dest, weight] : graph) {
        if (isGraph && weight < 0)
            continue;
        outfile << src << " " << dest << " " << weight << "\n";
    }
}
bool readMTXToTransposeCSR(const std::string& filename, 
                           std::vector<int>& values, 
                           std::vector<int>& row_indices, 
                           std::vector<int>& col_pointers, int flag = 0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return false;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    std::multimap<int, std::pair<int, int>> transposedEntries;

    int row, col, val;
    for (int i = 0; i < numNonZero; ++i) {
        file >> row >> col >> val;
        row--; // Convert to 0-based indexing
        col--;
        if ( flag == 0)
            transposedEntries.insert({col, {row, val}});
        else if (flag == 1 && val >= 0)
            transposedEntries.insert({col, {row, val}});
        else if (flag == 2 && val < 0)
            transposedEntries.insert({col, {row, val}});
    }

    file.close();

    values.clear();
    row_indices.clear();
    col_pointers.clear();
    col_pointers.clear();

    int current_col = -1;
    for(const auto& entry: transposedEntries) {
        int col = entry.first;
        int row = entry.second.first;
        int value = entry.second.second;

        if(col > current_col) {
            for(int i = 0; i < (col - current_col); ++i) 
                col_pointers.push_back(values.size());
            current_col = col;
        }

        values.push_back(value);
        row_indices.push_back(row);
    }

    col_pointers.push_back(values.size()); // The last element of col_pointers
    return true;
}
std::vector<std::tuple<int, int, int>> generateChangedGraph(
    const std::vector<std::tuple<int, int, int>>& originalGraph,
    int numVertices,
    int numChanges,
    int minWeight,
    int maxWeight,
    std::vector<std::tuple<int, int, int>>& changedEdges,
    float deletionPercentage) {
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
//Done



// Need to check
bool neighborConvertToTransposeCSR(const std::vector<std::vector<int>>& edges,
                                   std::vector<int>& values,
                                   std::vector<int>& row_indices,
                                   std::vector<int>& col_pointers) {
    std::multimap<int, std::pair<int, int>> transposedEntries;
    for (const auto& edge : edges) {
        int row = edge[0]; // source is row
        int col = edge[1]; // destination is col
        int val = edge[2]; // weight is val
        transposedEntries.insert({col, {row, val}});
    }

    int numCols = 0;
    for (const auto& entry: transposedEntries) {
        numCols = std::max(numCols, entry.first + 1);
    }

    values.clear();
    row_indices.clear();
    col_pointers.clear();

    // Initialize col_pointers with zeros
    col_pointers.resize(numCols + 1, 0);

    int current_col = -1;
    for (const auto& entry: transposedEntries) {
        int col = entry.first;
        int row = entry.second.first;
        int value = entry.second.second;

        if (col > current_col) {
            for (int i = current_col + 1; i <= col; ++i)
                col_pointers[i] = values.size();
            current_col = col;
        }

        values.push_back(value);
        row_indices.push_back(row);
    }
    col_pointers[numCols] = values.size(); // The last element of col_pointers
    return true;
}
std::vector<std::vector<int>> find_outgoing_connections(
    const std::vector<int> &values, 
    const std::vector<int> &row_ptr, 
    const std::vector<int> &col_idx, 
    const std::unordered_set<int> &vertices) {
    std::vector<std::vector<int>> outgoing_connections;
    for (const int vertex : vertices) {
        int start = row_ptr[vertex];
        int end = row_ptr[vertex + 1];

        for (int i = start; i < end; ++i) {
            int adjacent_vertex = col_idx[i];
            int weight = values[i];
            outgoing_connections.push_back({vertex, adjacent_vertex, weight});
        }
    }
    return outgoing_connections;
}

// Need to remove recursion
void markSubtreeAffected(const std::vector<int>& sssp_values, 
                         const std::vector<int>& sssp_column_indices, 
                         const std::vector<int>& sssp_row_pointers, 
                         std::vector<int>& dist, 
                         std::vector<bool>& isAffectedForDeletion, 
                         std::queue<int>& affectedNodesForDeletion, 
                         int node) {

    dist[node] = INT_MAX; // Invalidate the shortest distance
    isAffectedForDeletion[node] = true;
    affectedNodesForDeletion.push(node);
  
    // Get the start and end pointers for the row in CSR representation
    int start = sssp_row_pointers[node]; // Already 1-indexed
    int end = sssp_row_pointers[node + 1]; // Already 1-indexed

    // Traverse the CSR to find the children of the current node
    for (int i = start; i < end; ++i) {
        int child = sssp_column_indices[i]; // Already 1-indexed
        //std::cout<< child << " "<<std::endl;
        // If this child node is not already marked as affected, call the function recursively
        if (!isAffectedForDeletion[child]) {
            markSubtreeAffected(sssp_values, sssp_column_indices, sssp_row_pointers, dist, isAffectedForDeletion, affectedNodesForDeletion, child);
        }
    }
}


/*
1. ssspTree (done - regular 0-indexed)
2. graphCSR (done -regular 0-indexed)
3. shortestDist (done 0-indexed)
4. parentList (parent 0-indexed)
5. Predecessor (done - transposed 0-indexed)
6. Changed edges (done - transposed 0-indexed)
*/


void updateShortestPath( std::vector<int>& new_graph_values,  std::vector<int>& new_graph_column_indices,  std::vector<int>& new_graph_row_pointers, 
                         std::vector<int>& sssp_values,  std::vector<int>& sssp_column_indices,  std::vector<int>& sssp_row_pointers,
                        std::vector<int>& dist, std::vector<int>& parent , std::vector<int>& inDegreeValues, std::vector<int>& inDegreeColumnPointers, std::vector<int>& inDegreeRowValues) {
    
    std::cout << "Distance Before" <<std::endl;
    for (int i = 0; i < dist.size(); i++) {
        std::cout <<dist[i]<< " ";
    }
    std::cout<<std::endl;

    std::cout << "Parent Before " <<std::endl;
    for (int i = 0; i < parent.size(); i++) {
        std::cout <<parent[i]<< " ";
    }
    std::cout<<std::endl;

    
    std::vector<int> t_insert_values, t_insert_row_indices, t_insert_column_pointers;
    readMTXToTransposeCSR("changed_edges.mtx", t_insert_values, t_insert_row_indices, t_insert_column_pointers, 1); // Insert mode 1
    std::vector<int> t_delete_values, t_delete_row_indices, t_delete_column_pointers;
    readMTXToTransposeCSR("changed_edges.mtx", t_delete_values, t_delete_row_indices, t_delete_column_pointers, 2); // Delete mode 2

    std::vector<int> affectedNodesList(sssp_row_pointers.size(), 0);
    std::vector<int> affectedNodesN(sssp_row_pointers.size(), 0);
    std::vector<int> affectedNodesDel(sssp_row_pointers.size(), 0);


    cl::sycl::queue q(cl::sycl::gpu_selector_v);

    // For insertion    
    {
        // Changed Edges
        cl::sycl::buffer t_insert_column_pointers_buf(t_insert_column_pointers.data(), cl::sycl::range<1>(t_insert_column_pointers.size()));
        cl::sycl::buffer t_insert_row_indices_buf(t_insert_row_indices.data(), cl::sycl::range<1>(t_insert_row_indices.size()));
        cl::sycl::buffer t_insert_values_buf(t_insert_values.data(), cl::sycl::range<1>(t_insert_values.size()));

        // SSSP Tree
        cl::sycl::buffer sssp_values_buf(sssp_values.data(), cl::sycl::range<1>(sssp_values.size()));
        cl::sycl::buffer sssp_column_indices_buf(sssp_column_indices.data(), cl::sycl::range<1>(sssp_column_indices.size()));
        cl::sycl::buffer sssp_row_pointers_buf(sssp_row_pointers.data(), cl::sycl::range<1>(sssp_row_pointers.size()));

        // Distance
        cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

        // Parent
        cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));
        
        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));


        q.submit([&](cl::sycl::handler& cgh) 
        {
            auto t_insert_column_pointers_acc = t_insert_column_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_insert_row_indices_acc = t_insert_row_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_insert_values_acc = t_insert_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto sssp_values_acc = sssp_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_column_indices_acc = sssp_column_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_row_pointers_acc = sssp_row_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
    
            
             cgh.parallel_for<class MyKernel>(sycl::range<1>{t_insert_column_pointers_acc.size() - 1}, [=](sycl::id<1> idx) 
             {  
                // Check if (v->u) improves or not.
                int u = idx[0];
                int start = t_insert_column_pointers_acc[u];
                int end = t_insert_column_pointers_acc[u + 1];

                for (int i = start; i < end; ++i) {
                    int v = t_insert_row_indices_acc[i];
                    affectedNodesList_acc[v] = 0;
                    int alt = dist_acc[v] + t_insert_values_acc[i];
                    if (alt < dist_acc[u]) {
                        dist_acc[u] = alt;
                        parent_acc[u] = v;
                        affectedNodesList_acc[u] = 1; 
                    }
                }
            });
        });
        q.wait_and_throw();

    }

    // For deletion
    {

        cl::sycl::buffer t_delete_column_pointers_buf(t_delete_column_pointers.data(), cl::sycl::range<1>(t_delete_column_pointers.size()));
        cl::sycl::buffer t_delete_row_indices_buf(t_delete_row_indices.data(), cl::sycl::range<1>(t_delete_row_indices.size()));
        cl::sycl::buffer t_delete_values_buf(t_delete_values.data(), cl::sycl::range<1>(t_delete_values.size()));

        // SSSP Tree
        cl::sycl::buffer sssp_values_buf(sssp_values.data(), cl::sycl::range<1>(sssp_values.size()));
        cl::sycl::buffer sssp_column_indices_buf(sssp_column_indices.data(), cl::sycl::range<1>(sssp_column_indices.size()));
        cl::sycl::buffer sssp_row_pointers_buf(sssp_row_pointers.data(), cl::sycl::range<1>(sssp_row_pointers.size()));

        cl::sycl::buffer inDegreeValues_buf(inDegreeValues.data(), cl::sycl::range<1>(inDegreeValues.size()));
        cl::sycl::buffer inDegreeColumnPointers_buf(inDegreeColumnPointers.data(), cl::sycl::range<1>(inDegreeColumnPointers.size()));
        cl::sycl::buffer inDegreeRowValues_buf(inDegreeRowValues.data(), cl::sycl::range<1>(inDegreeRowValues.size()));

        // Distance
        cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

        // Parent
        cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));

        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));

        // AffectedNodesDel
        sycl::buffer<int> affectedNodesDel_buf(affectedNodesDel.data(), sycl::range<1>(affectedNodesDel.size()));

        q.submit([&](cl::sycl::handler& cgh) 
        {
            auto t_delete_column_pointers_acc = t_delete_column_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_delete_row_indices_acc = t_delete_row_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_delete_values_acc = t_delete_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto sssp_values_acc = sssp_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_column_indices_acc = sssp_column_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_row_pointers_acc = sssp_row_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

            auto inDegreeValues_acc = inDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto inDegreeColumnPointers_acc = inDegreeColumnPointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto inDegreeRowValues_acc = inDegreeRowValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto affectedNodesDel_acc = affectedNodesDel_buf.get_access<sycl::access::mode::read_write>(cgh);
    
            
             cgh.parallel_for<class MyKernel2>(sycl::range<1>{t_delete_column_pointers_acc.size() - 1}, [=](sycl::id<1> idx) 
             {
                // if (v -> u) is deleted
                int u = idx[0];

                int start = t_delete_column_pointers_acc[u];
                int end = t_delete_column_pointers_acc[u + 1];


                for (int i = start; i < end; ++i) {
                    int v = t_delete_row_indices_acc[i];
                    affectedNodesDel_acc[v] = 1; 
                    if (parent_acc[u] == v) {                        
                        affectedNodesDel_acc[u] = 1; // Mark the starting node
                        affectedNodesList_acc[u] = 1; 

                        int newDistance = INT_MAX;
                        int newParentIndex = -1;

                        int start = inDegreeColumnPointers_acc[u];
                        int end = inDegreeColumnPointers_acc[u + 1];

                        for(int i = start; i < end; ++i) {
                            int pred = inDegreeColumnPointers_acc[i]; // This is the vertex having an edge to 'u'
                            int weight = inDegreeValues_acc[i]; // This is the weight of the edge from 'vertex' to 'u'
                            
                            if(dist_acc[pred] + weight < newDistance )
                            {
                                newDistance = dist_acc[pred] + weight;
                                newParentIndex = pred; 
                            }
                            
                        }
                        
                        int oldParent = parent_acc[u];
                        if (newParentIndex == -1)
                        {
                            parent_acc[u] = -1; 
                            dist_acc[u] = INT_MAX; 
                        }
                        else
                        {
                            dist_acc[u] = newDistance;
                            parent_acc[u] = newParentIndex;
                            affectedNodesDel_acc[u] = 1;
                        }
                        
                    }
                }

            });
        });
        q.wait_and_throw();
    }

    {



        // SSSP Tree
        cl::sycl::buffer sssp_values_buf(sssp_values.data(), cl::sycl::range<1>(sssp_values.size()));
        cl::sycl::buffer sssp_column_indices_buf(sssp_column_indices.data(), cl::sycl::range<1>(sssp_column_indices.size()));
        cl::sycl::buffer sssp_row_pointers_buf(sssp_row_pointers.data(), cl::sycl::range<1>(sssp_row_pointers.size()));

        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));
        sycl::buffer<int> affectedNodesN_buf(affectedNodesN.data(), sycl::range<1>(affectedNodesN.size()));


        q.submit([&](cl::sycl::handler& cgh) 
        {
            
            auto sssp_values_acc = sssp_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_column_indices_acc = sssp_column_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto sssp_row_pointers_acc = sssp_row_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto affectedNodesN_acc = affectedNodesN_buf.get_access<sycl::access::mode::read_write>(cgh);
            
             cgh.parallel_for<class MyKernel3>(sycl::range<1>{affectedNodesList_acc.size()}, [=](sycl::id<1> idx) 
             {
                int u = idx[0];
                int start = sssp_row_pointers_acc[u];
                int end = sssp_row_pointers_acc[u + 1];
                for (int i = start; i < end; ++i) {
                    int v = sssp_column_indices_acc[i];
                    affectedNodesN_acc[v] = 1; 
                }

            });
        });
        q.wait_and_throw();
    }



    while(1)
    {
        size_t n = affectedNodesN.size();
        int result = 0;
        cl::sycl::buffer<int, 1> buffer_vector(affectedNodesN.data(), cl::sycl::range<1>(n));
        cl::sycl::buffer<int, 1> buffer_result(&result, cl::sycl::range<1>(1));
        q.submit([&](cl::sycl::handler& cgh) {

            auto acc_vector = buffer_vector.get_access<cl::sycl::access::mode::read>(cgh);
            auto acc_result = buffer_result.get_access<cl::sycl::access::mode::write>(cgh);

            // Perform the reduction
            cgh.single_task<class vector_reduction>([=]() {
                int sum = 0;
                for (size_t i = 0; i < n; i++) {
                    sum += acc_vector[i];
                }
                acc_result[0] = sum;
            });
        });
        q.wait_and_throw();

        auto host_result = buffer_result.get_access<cl::sycl::access::mode::read>();
        int sum = host_result[0];

        //std::cout<<sum<<std::endl;

        if (!sum)
            break;
        
        // For insertion    
        {

            // SSSP Tree
            cl::sycl::buffer sssp_values_buf(sssp_values.data(), cl::sycl::range<1>(sssp_values.size()));
            cl::sycl::buffer sssp_column_indices_buf(sssp_column_indices.data(), cl::sycl::range<1>(sssp_column_indices.size()));
            cl::sycl::buffer sssp_row_pointers_buf(sssp_row_pointers.data(), cl::sycl::range<1>(sssp_row_pointers.size()));

            cl::sycl::buffer inDegreeValues_buf(inDegreeValues.data(), cl::sycl::range<1>(inDegreeValues.size()));
            cl::sycl::buffer inDegreeColumnPointers_buf(inDegreeColumnPointers.data(), cl::sycl::range<1>(inDegreeColumnPointers.size()));
            cl::sycl::buffer inDegreeRowValues_buf(inDegreeRowValues.data(), cl::sycl::range<1>(inDegreeRowValues.size()));

            // Distance
            cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

            // Parent
            cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));

            // AffectedNodesList
            sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));
            sycl::buffer<int> affectedNodesN_buf(affectedNodesN.data(), sycl::range<1>(affectedNodesN.size()));

            // AffectedNodesDel
            sycl::buffer<int> affectedNodesDel_buf(affectedNodesDel.data(), sycl::range<1>(affectedNodesDel.size()));


            q.submit([&](cl::sycl::handler& cgh) 
            {
                
                auto sssp_values_acc = sssp_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto sssp_column_indices_acc = sssp_column_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto sssp_row_pointers_acc = sssp_row_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                
                auto inDegreeValues_acc = inDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto inDegreeColumnPointers_acc = inDegreeColumnPointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto inDegreeRowValues_acc = inDegreeRowValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                
                auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


                auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto affectedNodesN_acc = affectedNodesN_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto affectedNodesDel_acc = affectedNodesDel_buf.get_access<sycl::access::mode::read_write>(cgh);
        
                
                cgh.parallel_for<class MyKernel4>(sycl::range<1>{affectedNodesN_acc.size()}, [=](sycl::id<1> idx) 
                {  
                    //propagate [A(v)->n]
                    int n = idx[0];
                    if(affectedNodesN_acc[n] == 1)
                    {
                        affectedNodesN_acc[n] = 0; 
                        int start = inDegreeColumnPointers_acc[n];
                        int end = inDegreeColumnPointers_acc[n + 1];


                        for (int i = start; i < end; ++i) {
                            int v = inDegreeRowValues_acc[i];
                            if (affectedNodesList_acc[v] == 1)
                            {
                                affectedNodesList_acc[v] = 0;
                                if (affectedNodesDel_acc[v] == 1)
                                {
                                    affectedNodesDel_acc[v] = 0; 
                                    if (n + 1 == 1)
                                        continue;
                                    int newDistance = INT_MAX;
                                    int newParentIndex = -1;

                                    int start = inDegreeColumnPointers_acc[n];
                                    int end = inDegreeColumnPointers_acc[n + 1];

                                    for(int j = start; j < end; ++j) {
                                        int pred = inDegreeColumnPointers_acc[j]; // This is the vertex having an edge to 'u'
                                        int weight = inDegreeValues_acc[j]; // This is the weight of the edge from 'vertex' to 'u'
                                        
                                        if(dist_acc[pred] + weight < newDistance )
                                        {
                                            newDistance = dist_acc[pred] + weight;
                                            newParentIndex = pred; 
                                        }
                                        
                                    }
                                    
                                    int oldParent = parent_acc[n];
                                    if (newParentIndex == -1)
                                    {
                                        parent_acc[n] = -1; 
                                        dist_acc[n] = INT_MAX; 
                                    }
                                    else
                                    {
                                        dist_acc[n] = newDistance;
                                        parent_acc[n] = newParentIndex;
                                        affectedNodesDel_acc[n] = 1;
                                    }
                                }
                                int w = inDegreeValues_acc[n];
                                int start = inDegreeColumnPointers_acc[n];
                                int end = inDegreeColumnPointers_acc[n + 1];

                                for (int k = start; k < end; ++k) {
                                    int v = inDegreeRowValues_acc[k];
                                    affectedNodesList_acc[v] = 0;
                                    int alt = dist_acc[v] + inDegreeRowValues_acc[i];
                                    if (alt < dist_acc[n]) {
                                        dist_acc[n] = alt;
                                        parent_acc[n] = v;
                                        affectedNodesList_acc[n] = 1; 
                                    }
                                }
                            }
                        }
                    }

                });
            });
            q.wait_and_throw();
        }
    }

}


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

    // Add the last row_pointer, why?
    row_pointers[current_row + 1] = nnz;

    // Close the file
    file.close();

    return true;
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
    // road-road-usa.mtx, rgg_n_2_20_s0.mtx, 
    sortAndSaveMTX("graph.mtx", "sorted_graph.mtx");

    std::vector<int> values;
    std::vector<int> column_indices;
    std::vector<int> row_pointers;

    readMTXToCSR("sorted_graph.mtx", values, column_indices, row_pointers);
    //printCSRRepresentation(values, column_indices, row_pointers);

//Find SSSP tree and store in mtx file
    std::vector<int> parent(row_pointers.size() - 1, -1); // takes child and returns it's parent
    std::vector<int> dist(row_pointers.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values, column_indices, row_pointers, 0, dist, parent);

    saveSSSPTreeToFile("SSSP_Tree.mtx", values, column_indices, row_pointers, parent);

    std::vector<int> sssp_values;
    std::vector<int> sssp_column_indices;
    std::vector<int> sssp_row_pointers;
    sortAndSaveMTX("SSSP_Tree.mtx", "sorted_SSSP_Tree.mtx");
    readMTXToCSR("sorted_SSSP_Tree.mtx", sssp_values, sssp_column_indices, sssp_row_pointers);
    //printCSRRepresentation(sssp_values, sssp_column_indices, sssp_row_pointers);
// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    int numVertices = row_pointers.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges = 1;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    float deletePercentage = 1.0f;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges, deletePercentage);
    // writeMTX by default sort by row for easy readings
    writeMTX("new_graph.mtx", newGraph, numVertices, true); 
    writeMTX("changed_edges.mtx", changedEdges, numVertices, false);

    std::vector<int> new_graph_values;
    std::vector<int> new_graph_column_indices;
    std::vector<int> new_graph_row_pointers;
    readMTXToCSR("new_graph.mtx", new_graph_values, new_graph_column_indices, new_graph_row_pointers);
    //printCSRRepresentation(new_graph_values, new_graph_column_indices, new_graph_row_pointers);


    // Find Predecessor
    std::vector<int> inDegreeValues;
    std::vector<int> inDegreeColumnPointers;
    std::vector<int> inDegreeRowValues;
    readMTXToTransposeCSR("new_graph.mtx", inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);
    //printCSRRepresentation(inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);

    

    updateShortestPath(new_graph_values, new_graph_column_indices, new_graph_row_pointers, sssp_values, sssp_column_indices, sssp_row_pointers, dist, parent, inDegreeValues, inDegreeColumnPointers, inDegreeRowValues);

    return 0;
}

//clang++ -std=c++17 mtx2CSR.cpp  && ./a.out
