using namespace std;
#include <iostream> 
#include <omp.h>
#include <bits/stdc++.h>
#include <cstdlib>
#include <time.h>
#include <hdf5.h>

void print_array(float *array, int length){
    for (int i = 0; i < length; i++)
    {
        cout << array[i] << "\t";
    }
    cout << "\n";
}

void bitonicSort(int start, int end, int asc, float *array) {
    int num_swaps = 0; 
    int N = end - start + 1;
        
    for (int j = N/2; j > 0; j = j/2){         // steps in stages 
        num_swaps = 0;
        for (int i = start; i + j <= end; i++){
            if (num_swaps < j){
                if (asc){                       // sort ascendingly
                    if (array[i+j] < array[i]){
                        float temp = array[i+j];
                        array[i+j] = array[i];
                        array[i] = temp;
                    }
                        
                }
                     
                else{                           // sort descending
                    if (array[i+j] > array[i]){
                        float temp = array[i+j];
                        array[i+j] = array[i];
                        array[i] = temp;
                    }
                }

                num_swaps++;
            }
            else{
                num_swaps = 0;
                i = i + j - 1;
            }
        }
    }
}

bool verification(float *array, int length){
    if(length ==1) return true;
    if(length ==0) return true;
    if(array[length-2] > array[length-1]) return false;

    return verification(array, length-1);
}

int main() 
{
    int N;
    bool asc;
    int thread_num;
    double start, stop;
    //cout << "Enter the number of elements to be sorted: ";
    cin >> N;
    srand(time(NULL));
    float a = 10.0;

    hid_t file_id;
    herr_t status;
    
    file_id = H5Fcreate("openMP_input.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);  // hdf5 file creation

    hsize_t dims[1];
    dims[0] = N;            // size of array to be sorted
    float dset[dims[0]];    // declare array to write in hdf5 file
       

    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "/dset", H5T_IEEE_F32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    for(int i=0; i<N;i++){
        dset[i] = float((rand())/float((RAND_MAX)) * a);  // initialize data with random floating numbers
    }

    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset); // write data to file

    float array[dims[0]];  // array to be sorted
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);  // read data from file
    
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Fclose(file_id);
    
    //cout << "Array before sorting" << endl;
    //print_array(array, N);

    
    
        thread_num = 64;
        start = omp_get_wtime();
        for (int i = 2; i <= N; i = i * 2){                    // dividing problem into stages
            #pragma omp parallel for num_threads(thread_num)   // passing number of threads using num_threads
            for (int j = 0; j < N; j = j + i){
                if (((j / i) % 2) == 0){                      // check if it needs to be in ascending order or descending
                    asc = true;
                    bitonicSort(j, j + i - 1, asc, array);
                }
                else{
                    asc = false;
                    bitonicSort(j, j + i - 1, asc, array);
                }
            }
        }
        stop = omp_get_wtime();
        cout << "Time taken for array size " << N << " with " << thread_num << " threads: " << stop-start << endl;
    
    //cout << "Array after sorting" << endl;
    //print_array(array, N);

    bool verify = verification(array, N);
    if(verify) cout << "Verification test - Array is sorted" << endl;
    else cout<< "Verification test - Array is not sorted" << endl;

    file_id = H5Fcreate("openMP_output.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t data_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "/dset", H5T_IEEE_F32BE, data_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    
}