#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

void sequentialBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n; i++)
    {
#pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void merge(vector<int> &arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;
    vector<int> L(arr.begin() + l, arr.begin() + m + 1);
    vector<int> R(arr.begin() + m + 1, arr.begin() + r + 1);
    int i = 0;
    int j = 0;
    int k = l;
    while (i < n1 && j < n2)
    {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1)
    {
        arr[k++] = L[i++];
    }
    while (j < n2)
    {
        arr[k++] = R[j++];
    }
}

void sequentialMergeSort(vector<int> &arr, int l, int r)
{
    if (l < r)
    {
        int m = (l + r) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int> &arr, int l, int r, int depth = 0)
{
    if (l < r)
    {
        int m = (l + r) / 2;
        if (depth <= 4)
        {
#pragma omp parallel sections
            {
#pragma omp section
                parallelMergeSort(arr, l, m, depth + 1);
#pragma omp section
                parallelMergeSort(arr, m + 1, r, depth + 1);
            }
        }
        else
        {
            sequentialMergeSort(arr, l, m);
            sequentialMergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main()
{
    int N = 10000;
    vector<int> original(N);
    for (int i = 0; i < N; ++i)
    {
        original[i] = rand() % 100000;
    }
    vector<int> arr;
    arr = original;
    double start = omp_get_wtime();
    sequentialBubbleSort(arr);
    double end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds\n";
    arr = original;
    start = omp_get_wtime();
    parallelBubbleSort(arr);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";
    arr = original;
    start = omp_get_wtime();
    sequentialMergeSort(arr, 0, arr.size() - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << " seconds\n";
    arr = original;
    start = omp_get_wtime();
    parallelMergeSort(arr, 0, arr.size() - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";
    return 0;
}