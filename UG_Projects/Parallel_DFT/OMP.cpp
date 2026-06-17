#include<bits/stdc++.h>
using namespace std;

#define thread_count 8

typedef complex<double> Complex;

vector<vector<Complex>> loadSignalFromCSV(const string& filename) {
    vector<vector<Complex>> signal;
    ifstream inputFile(filename);
    if (inputFile.is_open()) {
        string line;
        while (getline(inputFile, line)) {
            vector<Complex> row;
            stringstream ss(line);
            string value;
            while (getline(ss, value, ',')) {
                row.push_back(stod(value));
            }
            signal.push_back(row);
        }
        inputFile.close();
    } else {
        cout << "Failed to open the input file" << endl;
    }
    return signal;
}

vector<Complex> dft(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> X(N, 0);

    #pragma omp parallel for num_threads(thread_count)
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            X[k] += x[n] * exp(Complex(0, -2 * M_PI * k * n / N));
        }
    }

    return X;
}

vector<vector<Complex>> dft2D(const vector<vector<Complex>>& x) {
    int M = x.size();
    int N = x[0].size();
    vector<vector<Complex>> X(M, vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < M; ++i) {
        X[i] = dft(x[i]);
    }

    vector<vector<Complex>> result(M, vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
    for (int j = 0; j < N; ++j) {
        vector<Complex> column;
        for (int i = 0; i < M; ++i) {
            column.push_back(X[i][j]);
        }
        column = dft(column);
        for (int i = 0; i < M; ++i) {
            result[i][j] = column[i];
        }
    }
    return result;
}

vector<Complex> idft(const vector<Complex>& X) {
    int N = X.size();
    vector<Complex> x(N, 0);

    #pragma omp parallel for num_threads(thread_count)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < N; ++k) {
            x[n] += X[k] * exp(Complex(0, 2 * M_PI * k * n / N)) ;
        }
        x[n] /= N;
    }

    return x;
}

vector<vector<Complex>> idft2D(const vector<vector<Complex>>& X) {
    int M = X.size();
    int N = X[0].size();
    vector<vector<Complex>> x(M, vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < M; ++i) {
        x[i] = idft(X[i]);
    }

    vector<vector<Complex>> result(M, vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
    for (int j = 0; j < N; ++j) {
        vector<Complex> column;
        for (int i = 0; i < M; ++i) {
            column.push_back(x[i][j]);
        }
        column = idft(column);
        for (int i = 0; i < M; ++i) {
            result[i][j] = column[i];
        }
    }
    return result;
}

void SaveCSV(const vector<vector<Complex>>& signal, const string& filename) {
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const auto& row : signal) {
            for (size_t i = 0; i < row.size(); ++i) {
                outputFile << row[i].real();
                if (i != row.size() - 1) {
                    outputFile << ",";
                }
            }
            outputFile << endl;
        }
        outputFile.close();
    } else {
        cout << "Failed to open the output file" << endl;
    }
}

int main(){

    //////////////////////////////////////////////////////////////////
    
    vector<vector<Complex>> signal = loadSignalFromCSV("OMP_signal.csv");
    double start, end;

    /////////////////////////////////////////////////////////////////

    start = clock();
    vector<vector<Complex>> dft_signal = dft2D(signal);
    end = clock();

    cout << "Time taken for 2D DFT: " << (end - start) / CLOCKS_PER_SEC << "s" << endl;
    
    ////////////////////////////////////////////////////////////////

    start = clock();
    vector<vector<Complex>> idft_signal = idft2D(dft_signal);
    end = clock();

    cout << "Time taken for 2D IDFT: " << (end - start) / CLOCKS_PER_SEC << "s" << endl;

    SaveCSV(idft_signal, "reconstructed_signal.csv");

    ///////////////////////////////////////////////////////////////

    /*
    Save the result to a CSV file
    saveSignalToCSV(dft_signal, "OMP_result.csv");
    saveSignalToCSV(idft_signal, "OMP_reconstructed_signal.csv");
    */

    ///////////////////////////////////////////////////////////////

}