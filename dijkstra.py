
import sys
import queue
import heapq



class PriorityQueue:
    def __init__(self):
        self.queue=[]
        self.index=0
        self.nodes=[]

    def push(self, item,priority):
                
        heapq.heappush(self.queue,(-priority,self.index,item))
        self.index=1

    def pop(self):
        return heapq.heappop(self.queue)[-1]




def distance(adj, cost, s, t):
    dist=[float('inf')]*len(adj)
    prev=[None]*len(adj)
    dist[s]=0
    used=[0]*len(adj)
    H=PriorityQueue()
    for i in range(len(adj)):
        H.push(i,-dist[i])
    while len(H.queue)>0:
        u=H.pop()
        if used[u]==0:
            used[u]==1
            j=0
            for v in adj[u]:
                if dist[v]>dist[u]+cost[u][j]:
                    dist[v]=dist[u]+cost[u][j]
                    prev[u]=v
                    H.push(v,-dist[v])
                j=j+1


    if dist[t]<float('inf'):
        return dist[t]
    else:
        return -1


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(zip(data[0:(3 * m):3], data[1:(3 * m):3]), data[2:(3 * m):3]))
    data = data[3 * m:]
    adj = [[] for _ in range(n)]
    cost = [[] for _ in range(n)]
    for ((a, b), w) in edges:
        adj[a - 1].append(b - 1)
        cost[a - 1].append(w)
    s, t = data[0] - 1, data[1] - 1
    print(distance(adj, cost, s, t))
