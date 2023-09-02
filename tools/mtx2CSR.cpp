#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void find_neighbors(const std::vector<int>& row_pointers, const std::vector<int>& column_indices, int vertex) {
    int start = row_pointers[vertex];
    int end = row_pointers[vertex + 1];
    std::cout << "Neighbors of vertex " << vertex << ": ";
    for (int i = start; i < end; ++i) {
        std::cout << column_indices[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::ifstream file("graph.mtx");

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

    // Test the function
    for (int vertex = 0; vertex < numRows; ++vertex) {
        find_neighbors(row_pointers, column_indices, vertex);
    }

    return 0;
}
