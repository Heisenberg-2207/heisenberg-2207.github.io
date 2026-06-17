2 project files are OMP.cpp and MPI.cpp and as the name says, they are the OpenMP and Open MPI implementation of the Discrete Fourier Transformation.
The OMP.cpp reads the file OMP_signal.csv file for performing computation and the MPI.cpp reads the MPI_signal.csv file for performing computation.
For OMP.cpp the python file helper.ipynb is used to convert a normal image to grayscale image and export as csv. 
The helper.ipynb also has few code blocks for reading the converted signal and checking if conversion was correct or not and also to visualise the conversions.
For MPI.cpp an array of size N*N and random values between 0-255 constructed using the same helper.ipynb notebook (N divisible by nprocs)
For plotting the results Plot.ipynb notebook is used. 
The OMP.cpp and MPI.cpp were run 10 times for each experiment conditions and the data was averaged and entered into the notebook manually.