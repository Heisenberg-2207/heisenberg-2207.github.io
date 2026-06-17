#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <complex>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;

    // Read array dimensions from CSV file
    if (rank == 0) {
        std::ifstream file("MPI_signal.csv");
        if (file.is_open()) {
            std::string line;
            rows = 0;
            cols = 0;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                int col = 0;
                while (std::getline(ss, cell, ',')) {
                    col++;
                }
                rows++;
                cols = col;
            }
            file.close();
        } else {
            std::cout << "Failed to open MPI_signal.csv" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast array dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create a 2D array of complex numbers
    std::complex<double> array[rows][cols];

    // Read array from CSV file
    if (rank == 0) {
        std::ifstream file("MPI_signal.csv");
        if (file.is_open()) {
            std::string line;
            int row = 0;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                int col = 0;
                while (std::getline(ss, cell, ',')) {
                    array[row][col] = std::complex<double>(std::stod(cell), 0.0);
                    col++;
                }
                row++;
            }
            file.close();
        } else {
            std::cout << "Failed to open MPI_signal.csv" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    double start, end;
    start = MPI_Wtime();

    // Perform row-wise decomposition
    int rows_per_process = rows / size;
    std::complex<double> local_rows[rows_per_process][cols];

    MPI_Scatter(&array[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &local_rows[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    std::complex<double> dft[rows][cols], local_dft[rows_per_process][cols];
    for (int i = 0; i < rows_per_process; i++) {
        for (int k = 0; k < cols; k++) {
            local_dft[i][k] = 0.0;
            for (int n = 0; n < cols; n++) {
                double angle = 2 * M_PI * k * n / cols;
                std::complex<double> complex_angle(cos(angle), -sin(angle));
                local_dft[i][k] += local_rows[i][n] * complex_angle;
            }
        }
    }

    MPI_Gather(&local_dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);


    std::complex<double> transposed_dft[cols][rows];
    if(rank == 0){
        // Transpose the DFT matrix
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed_dft[j][i] = dft[i][j];
            }
        }
    }

    /////////////////////////////////////////////////////////////

    std::complex<double> local_rows2[rows_per_process][cols];

    MPI_Scatter(&transposed_dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &local_rows2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    std::complex<double> transposed_dft2[rows][cols], local_dft2[rows_per_process][cols];
    for (int i = 0; i < rows_per_process; i++) {
        for (int k = 0; k < cols; k++) {
            local_dft2[i][k] = 0.0;
            for (int n = 0; n < cols; n++) {
                double angle = 2 * M_PI * k * n / cols;
                std::complex<double> complex_angle(cos(angle), -sin(angle));
                local_dft2[i][k] += local_rows2[i][n] * complex_angle;
            }
        }
    }

    MPI_Gather(&local_dft2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &transposed_dft2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    
    end = MPI_Wtime();
    
    std::complex<double> dft2[cols][rows];
    if(rank == 0){

        // Transpose the DFT matrix once more
        std::complex<double> transposed_dft[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dft2[j][i] = transposed_dft2[i][j];
            }
        }
        
        std::cout << "Time taken: " << end - start << " seconds" << std::endl;
        
        /*
        Save the result to a CSV file
        std::ofstream output_file("MPI_result.csv");
        if (output_file.is_open()) {
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    output_file << dft2[i][j].real() << "," << dft2[i][j].imag() << ",";
                }
                output_file << std::endl;
            }
            output_file.close();
        } else {
            std::cout << "Failed to open MPI_result.csv" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        */
    }

    
    MPI_Finalize();

    return 0;
}
