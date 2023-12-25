import math
import random
import heapq

CLOSED = "\033[93m" + "X" + "\033[0m"
OPEN = "\033[92m" +"O" + "\033[0m"
LEAK = "\033[96m" + "L" + "\033[0m"
BOT = "\033[94m" + "B" + "\033[0m"
TRACE = "\033[97m" + "O" + "\033[0m"

class Bot:
    def __init__(self, currIndex):
        self.currIndex = currIndex
        self.succeeded = False # Boolean variable to indicate whether the first leak is found

    def success(self):
        self.succeeded = True
    
    # Changes the bot index to the new index and checks if that newIndex is the leak position or not
    def step(self, graph, newIndex):    
        if graph[newIndex] == LEAK:
            self.success()
        graph[newIndex] = BOT
        graph[self.currIndex] = OPEN
        self.currIndex = newIndex
        return graph

    # Used for traversing the graph, make sure bot is only traveling to spaces that are open
    def getAdj(self, graph, n, index):
        up = index - n
        down = index + n
        left = index - 1
        right = index + 1
        
        possible = []
        
        if up >= 0:
            if graph[up] == OPEN or graph[up] == LEAK:
                possible.append(up)
            
        if down < n*n:
            if graph[down] == OPEN or graph[down] == LEAK:
                possible.append(down)
            
        if index%n != 0:
            if graph[left] == OPEN or graph[left] == LEAK:
                possible.append(left)
        
        if (index+1)%n != 0:
            if graph[right] == OPEN or graph[right] == LEAK:
                possible.append(right)
                
        return possible

    # A method to find the closest node in the given array 
    def find_closest_node(self, graph, n, may_contain_leak):
        queue = []
        heapq.heappush(queue, (0, self.currIndex)) # Use a priority queue to keep track of the distances 
        visited = set() 

        closest_nodes = []
        closest_distance = float('inf')

        # BFS loop
        while queue:
            distance, node = heapq.heappop(queue) # Pop the node with the smallest distance

            # Skip if this one is already been visited
            if node in visited:
                continue

            visited.add(node)

            # Check if the node is in the given array may_contain_leak and is less than or equal to the closest distance variable
            if node in may_contain_leak and distance <= closest_distance:
                # If the distance is smaller, we will create a new array of closest nodes and add this to the list
                # If the distance is the same, we will add it to current array of closest nodes
                if distance < closest_distance: 
                    closest_nodes = []
                closest_nodes.append(node)
                closest_distance = distance

            # Explore the neighbors of the current node
            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in visited:
                  heapq.heappush(queue, (distance + 1, neighbor)) # Add unvisited neighbors to the queue with updated distance
        
        # From our array of closest nodes, randomly choose one
        closest_node = random.choice(closest_nodes)
        return closest_node, closest_distance

    
    def check_detection_radius(self, graph, n, radius, may_contain_leak, leakIndex):
        botX = self.currIndex % n
        botY = self.currIndex // n
        detection_range = set()
        for x in range(botX - radius, botX + radius + 1):
            for y in range(botY - radius, botY + radius + 1):
                if 0 <= x < n and 0 <= y < n:
                    if (y * n + x) in may_contain_leak:
                        detection_range.add(y * n + x)

        leak_found = False
        if leakIndex in detection_range:
            leak_found = True
        return detection_range, leak_found

class Bot1(Bot):
    pass

class Bot2(Bot):
    # Same method as the find_closest_best_node from the initial bot, except with slight modifications
    def find_closest_best_node(self, graph, n, radius, may_contain_leak):
        queue = []
        heapq.heappush(queue, (0, self.currIndex))
        visited = set() 

        closest_nodes = []
        closest_distance = float('inf')

        while queue:
            distance, node = heapq.heappop(queue)

            if node in visited:
                continue

            visited.add(node)

            if node in may_contain_leak and distance <= closest_distance:
                if distance < closest_distance:
                    closest_nodes = []
                closest_nodes.append(node)
                closest_distance = distance

            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in visited:
                  heapq.heappush(queue, (distance + 1, neighbor))
        
        # Instead of choosing a random node, we will now choose the node that has the most unvisited cells around it
        best_node = None
        mostUncheckedNodes = 0
        for node in closest_nodes:
            uncheckedNodes = Bot2.count_unchecked_nodes_in_radius(node, n, radius, may_contain_leak)
            if uncheckedNodes > mostUncheckedNodes:
                mostUncheckedNodes = uncheckedNodes
                best_node = node

        return best_node, closest_distance 
    
    # Method to count how many unvisited cells are around a given node 
    # Max unvisited cells would be 8
    # o o o
    # o X o
    # o o o
    def count_unchecked_nodes_in_radius(node, n, radius, may_contain_leak):
        posX = node % n
        posY = node // n
        detection_range = []
        for x in range(posX - radius, posX + radius + 1):
            for y in range(posY - radius, posY + radius + 1):
                if 0 <= x < n and 0 <= y < n:
                    if (y * n + x) in may_contain_leak:
                        detection_range.append(y * n + x)

        return len(detection_range)


class Bot3(Bot):
    def getAdjD(self, graph, n, index):
        up = index - n
        down = index + n
        left = index - 1
        right = index + 1
        
        possible = []
        
        if up >= 0:
            if graph[up] == OPEN or graph[up] == LEAK or graph[up] == BOT:
                possible.append(up)
            
        if down < n*n:
            if graph[down] == OPEN or graph[down] == LEAK or graph[down] == BOT:
                possible.append(down)
            
        if index%n != 0:
            if graph[left] == OPEN or graph[left] == LEAK or graph[left] == BOT: 
                possible.append(left)
        
        if (index+1)%n != 0:
            if graph[right] == OPEN or graph[right] == LEAK or graph[right] == BOT:
                possible.append(right)
                
        return possible
    
    def dijkstra(self, graph, n, i, OPEN_SQUARES):
        dist = {}
        q = []
        
        dist[i] = 0
        q.append(i)
        
        for index in OPEN_SQUARES:
            if index == i:
                continue
            dist[index] = 9999
            q.append(index)
        
        while len(q) != 0:
            q.sort(key=lambda x: dist.get(x))
            
            u = q.pop(0)
            uAdj = Bot3.getAdjD(self, graph, n, u)
            
            for v in uAdj:
                if dist.get(v) > dist.get(u) + 1:
                    dist[v] = dist[u] + 1
        return dist
    
    def initDistance(self, graph, n, OPEN_SQUARES):
        allDistances = {}
        for index in OPEN_SQUARES:
            allDistances[index] = Bot3.dijkstra(self, graph, n, index, OPEN_SQUARES)
        
        return allDistances
            
    def getDistance(self, allDistances, i, j):
        return allDistances.get(i).get(j)
    
    def getShortestPath(self, graph, n, start, target):
        queue = [start]
        parent = {start: None}
        while queue:
            node = queue.pop(0)  
            if node == target:
                path = []
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path))

            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    

    def initProbability(self, graph, n):
        count = 0
        
        for i in range(n*n):
            if graph[i] == CLOSED:
                count += 1
                
        m = n*n - count
        s = 1/m 
        
        prob = [0]*(n*n)
        
        for i in range(n*n):
            if graph[i] != CLOSED:
                prob[i] = s
        
        return prob
    
    def normalizeHeatMap(self, heatMap, n):
        total = 0
        for index in range(0, n*n):
            total += heatMap[index]

        if total == 0:
            return heatMap
        
        for index in range(0, n*n):
            heatMap[index] = (heatMap[index] / total)
        
        return heatMap
    
    def botEntersCellUpdate(self, prob, graph, n):
        adj = Bot.getAdj(self, graph, n, self.currIndex)
        pLeakFound = 0
        
        for index in adj:
            pLeakFound += prob[index]
        
        pLeakNotFound = 1- pLeakFound
        
        if pLeakNotFound + 0.0 == 0.0:
            return prob
        
        for i in range(0, n*n):
            pLeakNotFoundGivenLeakInJ = 1
            if graph[i] == CLOSED:
                continue
            if i in adj:
                pLeakNotFoundGivenLeakInJ = 0
                
            pLeakInJ = prob[i]
            
            prob[i] = pLeakInJ * pLeakNotFoundGivenLeakInJ / pLeakNotFound
        
        prob = Bot3.normalizeHeatMap(self, prob, n)
        return prob
            
    def noBeepUpdateProb(self, prob, pBeep, graph, n, a, allDist):
        totalProb = 0
        for x in range(len(prob)):
            if prob[x] > 0:
              d = Bot3.getDistance(self, allDist, self.currIndex, x)
              totalProb += prob[x] * (1 - math.exp(-a*(d-1)))

        for x in range(len(prob)):
            if prob[x] > 0:
                prob[x] = ( prob[x] * (1 - pBeep)) / totalProb
        
        return prob
        
    def beepUpdateProb(self, prob, pBeep, graph, n, a, allDist):
        totalProb = 0
        for x in range(len(prob)):
            if prob[x] > 0:
              d = Bot3.getDistance(self, allDist, self.currIndex, x)
              totalProb += prob[x] * math.exp(-a*(d-1))

        for x in range(len(prob)):
            if prob[x] > 0:
                prob[x] = ( prob[x] * pBeep) / totalProb
                
        return prob
    
    def beep(self, a, graph, n, leak, allDist):
        d = Bot3.getDistance(self, allDist, self.currIndex, leak)
        p = math.exp(-a*(d-1))
        rng = random.random()
        
        if rng <= p:
            return True, p 
        
        return False, p

    
    def getHighestProbabilities(self, probabilityMap):
        highestProbabilities = []
        highestProbability = 0
        for x in range(len(probabilityMap)):
            if probabilityMap[x] == highestProbability:
                highestProbabilities.append(x)
                
            if probabilityMap[x] > highestProbability:
                highestProbabilities = [x]
                highestProbability = probabilityMap[x]
        return highestProbabilities

    def moveUpdate(self, probabilityMap):
        sumProb = 0
        for x in probabilityMap:
            sumProb += x
        for x in range(len(probabilityMap)):
            if probabilityMap[x] != 0:
                probabilityMap[x] = probabilityMap[x] / sumProb
        return probabilityMap
    
    def replan(self, graph, n, heatMap):
        highestIndex = 0
        doubles = []
        
        for index in range(0, n*n):
            if heatMap[index] > heatMap[highestIndex]:
                doubles.clear()
                doubles.append(index)
                highestIndex = index
                
            if heatMap[index] == heatMap[highestIndex]:
                doubles.append(index)
            
        if len(doubles) > 1:
            highestIndex = random.choice(doubles)
        
        newPath = Bot3.getShortestPath(self, graph, n, self.currIndex, highestIndex)
        return newPath
    
    
class Bot4(Bot):
    def getShortestPath(self, graph, n, start, target):
        queue = [start] # Queue for BFS
        parent = {start: None} # Dictionary to store the parent nodes of each node in the shortest path
        while queue:
            node = queue.pop(0)  
            if node == target:
                path = []
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path)) # When node is found, we will return the array of nodes that make up the shortest path

            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    # This is the same method as from bot2
    def find_closest_best_node(self, graph, n, may_contain_leak, probMap):
        queue = []
        heapq.heappush(queue, (0, self.currIndex))
        visited = set() 

        closest_nodes = []
        closest_distance = float('inf')

        while queue:
            distance, node = heapq.heappop(queue)

            if node in visited:
                continue

            visited.add(node)

            if node in may_contain_leak and distance <= closest_distance:
                if distance < closest_distance:
                    closest_nodes = []
                closest_nodes.append(node)
                closest_distance = distance

            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in visited:
                  heapq.heappush(queue, (distance + 1, neighbor))
        
        # Like in bot2, we will choose the nodes that have the most unvisited cells around it
        best_node = None
        mostUncheckedNodes = 0
        for node in closest_nodes:
            uncheckedNodes = Bot4.count_unchecked_nodes(node, n, probMap)
            if uncheckedNodes > mostUncheckedNodes:
                mostUncheckedNodes = uncheckedNodes
                best_node = node

        return best_node, closest_distance 
    
    def count_unchecked_nodes(node, n, probMap):
        posX = node % n
        posY = node // n
        detection_range = []
        for x in range(posX - 1, posX + 2):
            for y in range(posY - 1, posY + 2):
                if 0 <= x < n and 0 <= y < n:
                    if probMap[(y * n + x)] > 0:
                        detection_range.append(y * n + x)

        return len(detection_range)
    
    # Method that returns the highest probabilities from our given probability map
    def getHighestProbabilities(self, probabilityMap):
        highestProbabilities = []
        highestProbability = 0
        for x in range(len(probabilityMap)):
            if probabilityMap[x] == highestProbability:
                highestProbabilities.append(x)
                
            if probabilityMap[x] > highestProbability:
                highestProbabilities = [x]
                highestProbability = probabilityMap[x]
        return highestProbabilities   
    
    # Method to change the probabilities of the map when the bot moves
    def moveUpdate(self, probabilityMap):
        sumProb = 0
        for x in probabilityMap:
            sumProb += x
        for x in range(len(probabilityMap)):
            if probabilityMap[x] > 0:
                probabilityMap[x] = probabilityMap[x] / sumProb
        return probabilityMap
    
    # Return the distance of the shortest path for given nodes
    def getDistance(self, graph, n, start, leak):
        trace = Bot4.getShortestPath(self, graph, n, start, leak)
        if trace == None:
            return -1
        return len(trace) - 1
    
    # Calculate the probability of the beep and using rng, return true/false
    def beep(self, a, graph, n, leak):
        d = Bot4.getDistance(self, graph, n, self.currIndex, leak)
        p = math.exp(-a*(d-1))
        rng = random.uniform(0, 1)
        
        if rng <= p:
            return True, p, d
        return False, p, d
    
    # Update the probabilities of the map given that there is no beep
    def beepUpdate(self, probMap, graph, n, a):
        totalProb = 0
        for x in range(len(probMap)):  
            if probMap[x] > 0:
              d = Bot4.getDistance(self, graph, n, self.currIndex, x)
              totalProb += probMap[x] * math.exp(-a*(d-1)) # Total probability of beep in current cell

        for x in range(len(probMap)):
            if probMap[x] > 0:
                d = Bot4.getDistance(self, graph, n, self.currIndex, x)
                # Update the probability of each cell to be P(leak in j) * P(beep in current cell | leak in j) / P(beep in current cell)
                probMap[x] = ( probMap[x] * math.exp(-a*(d-1))) / totalProb

        
        return probMap

    # Update the probabilities of the map given that there is no beep
    def noBeepUpdate(self, probMap, graph, n, a):
        totalProb = 0
        for x in range(len(probMap)):
            if probMap[x] > 0:
              d = Bot4.getDistance(self, graph, n, self.currIndex, x)
              totalProb += probMap[x] * (1 - math.exp(-a*(d-1))) # Total probability of no beep in current cell

        for x in range(len(probMap)):
            if probMap[x] > 0:
                d = Bot4.getDistance(self, graph, n, self.currIndex, x)
                # Update the probability of each cell to be P(leak in j) * P(no beep in current cell | leak in j) / P(no beep in current cell)
                probMap[x] = ( probMap[x] * (1 - math.exp(-a*(d-1)))) / totalProb
        
        return probMap
        
    
class Bot5(Bot):
    def __init__(self, currIndex):
        super().__init__(currIndex)
        self.succeeded2 = False # Boolean variable to indicate whether the second leak is found

    def success2(self):
        self.succeeded2 = True
    
    def step(self, graph, newIndex):    
        if graph[newIndex] == LEAK: # If we found the leak, check whether we mark the first leak as found or the second
            if self.succeeded == False: # If the first leak is not found, mark the first leak is found
                self.success()
            else: # If the first leak is found, mark the second leak is found
                self.success2()
        graph[newIndex] = BOT
        graph[self.currIndex] = OPEN
        self.currIndex = newIndex
        return graph
    
    # Method to check if there is a leak in the detection radius
    def check_detection_radius(self, graph, n, radius, may_contain_leak):
        botX = self.currIndex % n
        botY = self.currIndex // n
        detection_range = set()
        for x in range(botX - radius, botX + radius + 1):
            for y in range(botY - radius, botY + radius + 1):
                if 0 <= x < n and 0 <= y < n:
                    if (y * n + x) in may_contain_leak:
                        detection_range.add(y * n + x)

        leak_found = False
        for x in detection_range:
            if graph[x] == LEAK:
              leak_found = True
        return detection_range, leak_found
    
class Bot6(Bot5):
    # Similar to bot2 method, we find the closest best node that has the most unvisited cells around it
    def find_closest_best_node(self, graph, n, radius, may_contain_leak):
        queue = []
        heapq.heappush(queue, (0, self.currIndex))
        visited = set() 

        closest_nodes = []
        closest_distance = float('inf')

        while queue:
            distance, node = heapq.heappop(queue)

            if node in visited:
                continue

            visited.add(node)

            if node in may_contain_leak and distance <= closest_distance:
                if distance < closest_distance:
                    closest_nodes = []
                closest_nodes.append(node)
                closest_distance = distance

            for neighbor in Bot.getAdj(self, graph, n, node):
                if neighbor not in visited:
                  heapq.heappush(queue, (distance + 1, neighbor))
        
        best_node = None
        mostUncheckedNodes = 0
        for node in closest_nodes: # From our list of closest nodes, choose the node with the most unchecked cells around it
            uncheckedNodes = Bot6.count_unchecked_nodes_in_radius(node, n, radius, may_contain_leak)
            if uncheckedNodes > mostUncheckedNodes:
                mostUncheckedNodes = uncheckedNodes
                best_node = node

        return best_node, closest_distance 
    
    # Method to count the amount of unvisited cells around a node
    def count_unchecked_nodes_in_radius(node, n, radius, may_contain_leak):
        posX = node % n
        posY = node // n
        detection_range = []
        for x in range(posX - radius, posX + radius + 1):
            for y in range(posY - radius, posY + radius + 1):
                if 0 <= x < n and 0 <= y < n:
                    if (y * n + x) in may_contain_leak:
                        detection_range.append(y * n + x)

        return len(detection_range)
    
class Bot7(Bot3):
    def succeeded2(self):
        self.succeeded2 = True
        
class Bot8(Bot3):
    def beep2leaks(self, a, graph, n, leak1, leak2, allDist):
        beep1, pbeep1 =Bot3.beep(self, a, graph, n, leak1, allDist)
        beep2, pbeep2 =Bot3.beep(self, a, graph, n, leak2, allDist)
        
        pbeep = 1-((1 - pbeep1)*(1 - pbeep2))
        
        rng = random.random()
        
        if rng < pbeep:
            return True, pbeep, pbeep1, pbeep2
        
        return False, pbeep, pbeep1, pbeep2
    
    def noBeepUpdateProb(self, prob, pBeep, graph, n, a, allDist):
        totalProb = 0
        for x in range(len(prob)):
            for y in range(len(prob)):
                
                if prob[x] > 0 and prob[y] > 0:
                    dx = Bot3.getDistance(self, allDist, self.currIndex, x)
                    dy = Bot3.getDistance(self, allDist, self.currIndex, y)
                    totalProb += prob[x] * (1 - math.exp(-a*(dx-1))) * (1-math.exp(-a*(dy-1)))

        for x in range(len(prob)):
            if prob[x] > 0:
                prob[x] = ( prob[x] * (1 - pBeep)) / totalProb
        
        return prob
        
    def beepUpdateProb(self, prob, pBeep, graph, n, a, allDist):
        totalProb = 0
        for x in range(len(prob)):
            for y in range(len(prob)):
                
                if prob[x] > 0 and prob[y] > 0:
                    dx = Bot3.getDistance(self, allDist, self.currIndex, x)
                    dy = Bot3.getDistance(self, allDist, self.currIndex, y)
                    totalProb += prob[x] * (1 - math.exp(-a*(dx-1))) * (1-math.exp(-a*(dy-1)))

        for x in range(len(prob)):
            if prob[x] > 0:
                
                if totalProb == 0:
                    return prob 
                
                prob[x] = ( prob[x] * pBeep ) / totalProb
                
        return prob
    

class Bot9(Bot8):
    pass