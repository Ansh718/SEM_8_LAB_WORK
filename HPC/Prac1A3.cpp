#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void generateArray(vector<int> &arr, int size)
{
    srand(time(0));
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 1000;
    }
}

void sequentialOps(vector<int> &arr, int &miN, int &maX, long long &sum, double &avg)
{
    miN = INT_MAX;
    maX = INT_MIN;
    sum = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] < miN)
        {
            miN = arr[i];
        }
        if (arr[i] > maX)
        {
            maX = arr[i];
        }
        sum += arr[i];
    }
    avg = static_cast<double>(sum) / arr.size();
}

void parallelOps(vector<int> &arr, int &miN, int &maX, long long &sum, double &avg)
{
    miN = INT_MAX;
    maX = INT_MIN;
    sum = 0;
#pragma omp parallel for reduction(min : miN) reduction(max : maX) reduction(+ : sum)
    {
        for (int i = 0; i < arr.size(); i++)
        {
            if (arr[i] < miN)
            {
                miN = arr[i];
            }
            if (arr[i] > maX)
            {
                maX = arr[i];
            }
            sum += arr[i];
        }
    }
    avg = static_cast<double>(sum) / arr.size();
}

int main()
{
    int size = 1e7;
    vector<int> arr(size);
    generateArray(arr, size);
    int miNSeq, maXSeq, miNPar, maXPar;
    long long sumSeq, sumPar;
    double avgSeq, avgPar;
    double start = omp_get_wtime();
    sequentialOps(arr, miNSeq, maXSeq, sumSeq, avgSeq);
    double end = omp_get_wtime();
    double timeSeq = end - start;
    start = omp_get_wtime();
    parallelOps(arr, miNPar, maXPar, sumPar, avgPar);
    end = omp_get_wtime();
    double timePar = end - start;
    cout << "Sequential:\n";
    cout << "Min: " << miNSeq << ", Max: " << maXSeq
         << ", Sum: " << sumSeq << ", Avg: " << avgSeq << "\n";
    cout << "Time: " << timeSeq << " seconds\n\n";
    cout << "Parallel (OpenMP):\n";
    cout << "Min: " << miNPar << ", Max: " << maXPar
         << ", Sum: " << sumPar << ", Avg: " << avgPar << "\n";
    cout << "Time: " << timePar << " seconds\n";
    return 0;
}