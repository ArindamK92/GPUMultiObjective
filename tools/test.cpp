#include <iostream>
#include <vector>

using namespace std;

struct CSR {
    vector<int> values;
    vector<int> row_ptr;
    vector<int> col_idx;
};

vector<CSR> singleColumnCSR(const CSR& csr, int n_columns) {
    vector<CSR> singleColumnCSRs(n_columns);

    for (int i = 0; i < n_columns; ++i) {
        vector<int> values_i;
        vector<int> row_ptr_i = {0};
        vector<int> col_idx_i;

        for (int j = 0; j < csr.row_ptr.size() - 1; ++j) {
            int row_start = csr.row_ptr[j];
            int row_end = csr.row_ptr[j + 1];

            bool found = false;
            for (int k = row_start; k < row_end; ++k) {
                if (csr.col_idx[k] == i) {
                    values_i.push_back(csr.values[k]);
                    col_idx_i.push_back(0);
                    found = true;
                    break;
                }
            }

            if (found) {
                row_ptr_i.push_back(values_i.size());
            } else {
                row_ptr_i.push_back(row_ptr_i.back());
            }
        }

        singleColumnCSRs[i] = {values_i, row_ptr_i, col_idx_i};
    }

    return singleColumnCSRs;
}

CSR combineSingleColumnCSR(const vector<CSR>& singleColumnCSRs) {
    CSR original;
    int n_rows = singleColumnCSRs[0].row_ptr.size() - 1;

    original.row_ptr.push_back(0);

    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < singleColumnCSRs.size(); ++col) {
            const CSR& singleColumn = singleColumnCSRs[col];

            int row_start = singleColumn.row_ptr[row];
            int row_end = singleColumn.row_ptr[row + 1];

            for (int k = row_start; k < row_end; ++k) {
                original.values.push_back(singleColumn.values[k]);
                original.col_idx.push_back(col);
            }
        }
        original.row_ptr.push_back(original.values.size());
    }

    return original;
}

void printCSR(const CSR& csr) {
    cout << "Values: ";
    for (int val : csr.values) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Row Pointers: ";
    for (int ptr : csr.row_ptr) {
        cout << ptr << " ";
    }
    cout << endl;

    cout << "Column Indices: ";
    for (int idx : csr.col_idx) {
        cout << idx << " ";
    }
    cout << endl;
}

int main() {
    // Original CSR representation
    CSR original = {
        {1, 1, 1, 1, 1, 1, 1, 1},
        {0, 1, 4, 6, 8},
        {1, 0, 2, 3, 1, 3, 1, 2}
    };

    // Create single-column CSRs
    vector<CSR> singleColumnMatrices = singleColumnCSR(original, 4);

    // Print single-column CSRs
    for (int i = 0; i < singleColumnMatrices.size(); ++i) {
        cout << "Single-Column Matrix " << i + 1 << ":" << endl;
        printCSR(singleColumnMatrices[i]);
        cout << endl;
    }

    // Re-create the original CSR representation from the single-column CSR matrices
    CSR recombined = combineSingleColumnCSR(singleColumnMatrices);

    // Print the recombined CSR representation
    cout << "Recombined CSR Matrix:" << endl;
    printCSR(recombined);

    return 0;
}
