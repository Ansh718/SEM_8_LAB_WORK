#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;

class Graph
{
public:
    int vertices;
    vector<vector<int>> adj;

    Graph(int v) : vertices(v), adj(v) {}

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void generateLargeGraph(int edges)
    {
        srand(time(0));
        for (int i = 0; i < edges; ++i)
        {
            int u = rand() % vertices;
            int v = rand() % vertices;
            if (u != v)
            {
                addEdge(u, v);
            }
        }
    }

    void sequentialBFS(int start)
    {
        vector<bool> visited(vertices, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);
        while (!q.empty())
        {
            int node = q.front();
            q.pop();
            for (int neighbor : adj[node])
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    // Iterative (stack-based) DFS
    void sequentialDFS(int start)
    {
        vector<bool> visited(vertices, false);
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty())
        {
            int node = s.top();
            s.pop();

            for (int neighbor : adj[node])
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }

    void parallelBFS(int start)
    {
        vector<bool> visited(vertices, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);
        while (!q.empty())
        {
            int node;
#pragma omp critical
            {
                if (!q.empty())
                {
                    node = q.front();
                    q.pop();
                }
            }

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(adj[node].size()); ++i)
            {
                int neighbor = adj[node][i];
#pragma omp critical
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    // Parallel DFS using stack and parallelizing neighbors
    void parallelDFS(int start)
    {
        vector<bool> visited(vertices, false);
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty())
        {
            int node;
#pragma omp critical
            {
                node = s.top();
                s.pop();
            }

#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(adj[node].size()); ++i)
            {
                int neighbor = adj[node][i];
                bool shouldPush = false;

#pragma omp critical
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        shouldPush = true;
                    }
                }

                if (shouldPush)
                {
#pragma omp critical
                    s.push(neighbor);
                }
            }
        }
    }
};

int main()
{
    int v = 1000;
    int e = 5000;
    Graph g(v);
    g.generateLargeGraph(e);

    double start, end;

    start = omp_get_wtime();
    g.sequentialBFS(0);
    end = omp_get_wtime();
    cout << "Sequential BFS Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    g.parallelBFS(0);
    end = omp_get_wtime();
    cout << "Parallel BFS Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    g.sequentialDFS(0);
    end = omp_get_wtime();
    cout << "Sequential DFS Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    g.parallelDFS(0);
    end = omp_get_wtime();
    cout << "Parallel DFS Time: " << end - start << " seconds\n";

    return 0;
}