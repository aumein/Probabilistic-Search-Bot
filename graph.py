import random
import math
import bot
import time

CLOSED = "\033[93m" + "X" + "\033[0m"
OPEN = "\033[92m" +"O" + "\033[0m"
LEAK = "\033[96m" +"L" + "\033[0m"
BOT = "\033[94m" + "B" + "\033[0m"
TRACE = "\033[97m" + "O" + "\033[0m"

OPEN_SQUARES = []
ONE_NEIGHBOR = []
DEAD_ENDS = []

K = 1 
# (2k+1) x (2k+1)

DETECTION_SQUARES = []
MAY_CONTAIN_LEAK = []
PROBABILITY_MAP = []

#ALPHA = 1 # α > 0

#ONLY USED DURING INITIALIZATION
def openRandomInterior(graph,n):
    squareOpened = False
    
    if len(OPEN_SQUARES) == n*n:
        #all squares are open
        return
    
    while not squareOpened:
        a = random.randint(0, n*n-1)
        if a not in range(0, n-1) and a not in range(n*n-n, n*n-1) and a%n != 0 and (a+1)%n != 0:
            OPEN_SQUARES.append(a)
            graph[a] = OPEN
            squareOpened = True

    return

def openSquare(graph, index):
    if index == None:
        return
    
    OPEN_SQUARES.append(index)
    graph[index] = OPEN
    return
    

#Used to ID blocked cells with only one neighbor
def checkForOneAdj(n, index):
    up = index - n
    down = index + n
    left = index - 1
    right = index + 1
    
    count = 0
    
    if up >= 0:
        if up in OPEN_SQUARES:
            count+= 1
        
    if down < n*n:
        if down in OPEN_SQUARES:
            count+= 1
        
    if index%n != 0:
        if left in OPEN_SQUARES:
            count+= 1
    
    if (index+1)%n != 0:
        if right in OPEN_SQUARES:
            count+= 1  
    
    if count == 1:
        return True

    return False

#Actually only returns the index of an open adjacent square to index
#Used for the opening of dead ends
def openOneAdj(index, n):
    up = index - n
    down = index + n
    left = index - 1
    right = index + 1
    
    possible = []
    
    if up >= 0:
        if up not in OPEN_SQUARES:
            possible.append(up)
        
    if down < n*n:
        if down not in OPEN_SQUARES:
            possible.append(down)
        
    if index%n != 0:
        if left not in OPEN_SQUARES:
            possible.append(left)
    
    if (index+1)%n != 0:
        if right not in OPEN_SQUARES:
            possible.append(right)
            
    if len(possible) == 0:
        return None
    
    return random.choice(possible)

    
#Print Function
def printGraph(graph, n):
    print()
    for i in range(0,n*n):
        print(" " + str(graph[i]) + " ", end="")
        if((i+1)%n == 0 and i != 0):
            print()
    return

# Identify all currently blocked cells that have exactly one open neighbor.
# Of these currently blocked cells with exactly one open neighbor, pick one at random.
# Open the selected cell.
# Repeat until you can no longer do so.
def iterateOnInit(graph,n):
    ONE_NEIGHBOR = []
    
    for i in range(0, n*n):
        if checkForOneAdj(n, i) and i not in OPEN_SQUARES:
            ONE_NEIGHBOR.append(i)
    
    if len(ONE_NEIGHBOR) == 0:
        return -1
    
    index = random.choice(ONE_NEIGHBOR)
    openSquare(graph, index)
            
    return index

def deadEnds(n):
    DEAD_ENDS = []
    
    for i in range(0, n*n):
        if checkForOneAdj(n,i) and i in OPEN_SQUARES:
            DEAD_ENDS.append(i)
    
    if len(DEAD_ENDS) == 0:
        DEAD_ENDS.append(-1)

    return DEAD_ENDS

#Initializes a graph of size nxn
#Opens a random interior square to start the iteration function
#Opens about half of the dead ends that were found
def initGraph(n):
    size = n*n
    graph = [CLOSED]*size
    openRandomInterior(graph,n)
    index = 0
    
    #iteration
    while index != -1:
        index = iterateOnInit(graph, n)
    
    DEAD_ENDS = deadEnds(n) 
    
    toRev = math.trunc(0.5*len(DEAD_ENDS))
    
    for i in range(0, toRev):
        index = random.choice(DEAD_ENDS)
        openSquare(graph, openOneAdj(index, n))
        DEAD_ENDS.remove(index)
    
    return graph, n

# Initializes a grid of size n with the values of the initial probability
def initProbability(n, MAY_CONTAIN_LEAK):
    for x in range (n * n):
        PROBABILITY_MAP.append(0)
    for x in MAY_CONTAIN_LEAK:
        PROBABILITY_MAP[x] = 1.0 / (len(MAY_CONTAIN_LEAK)) # Initial Probability is 1 divided by the amount of open squares
    return PROBABILITY_MAP

# Print out the probability map
def printProbabilityMap(probMap, n):
    for i in range(n):
        for j in range(n):
            cell_index = i * n + j
            print(probMap[cell_index], end=" ")
        print("\n")

# Puts the bot onto the graph
def placeBot(graph, botIndex):
    graph[botIndex] = BOT

# If the bot runs on probability, place leak anywhere, otherwise place leak outside the detection radius
def placeLeak(graph, leakIndex, botIndex, n, probabilityOn, k_value):
    if probabilityOn:
        graph[leakIndex] = LEAK
    else:
      botX = botIndex % n
      botY = botIndex // n
      detection_range = [botIndex]
      for x in range(botX - k_value, botX + k_value + 1):
          for y in range(botY - k_value, botY + k_value + 1):
              if 0 <= x < n and 0 <= y < n:
                  detection_range.append(y * n + x)
      outside_detection_square = [i for i in OPEN_SQUARES if i not in detection_range]
      if outside_detection_square:
          leakIndex = random.choice(outside_detection_square)
          while graph[leakIndex] == LEAK:
              leakIndex = random.choice(outside_detection_square)
          graph[leakIndex] = LEAK
      else:
          # If the detection radius encompasses the whole grid, then just place the leak anywhere
          leakIndex = random.choice(OPEN_SQUARES)
          graph[leakIndex] = LEAK
    return leakIndex

def testBot1(k_value):
    graphSize = 30
    graph, n = initGraph(graphSize)
    botIndex = random.choice(OPEN_SQUARES)
    leakIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)
    leakIndex = placeLeak(graph, leakIndex, botIndex, n, False, k_value)
    MAY_CONTAIN_LEAK = set(OPEN_SQUARES)
    b1 = bot.Bot1(botIndex)
    actions = 0
    leak_found = False

    # Scan all the nodes in the detection radius
    detection_range, leak_found = b1.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK, leakIndex)
    actions += 1

    if not leak_found:
      # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
      MAY_CONTAIN_LEAK -= detection_range

    while not b1.succeeded:
        # If a leak is found, go through all the unchecked nodes in the detection range starting from the closest one
        if leak_found:
            closest_node, distance = b1.find_closest_node(graph, n, detection_range)
            actions += distance
            b1.step(graph, closest_node) # Move to the closest unchecked node in detection range
            if(b1.succeeded):
                break
            else:
                detection_range.remove(b1.currIndex) # If the space is not a leak, remove it
        else:
            # If a leak is not found, go to the closest node in MAY_CONTAIN_LEAK
            closest_node, distance = b1.find_closest_node(graph, n, MAY_CONTAIN_LEAK)
            actions += distance
            b1.step(graph, closest_node) # Move to the closest node in MAY_CONTAIN_LEAK
            if b1.succeeded:
                break
            else: # Scan all the nodes in the detection radius if bot1 hasn't landed on the leak position
                MAY_CONTAIN_LEAK.remove(b1.currIndex)
                detection_range, leak_found = b1.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK, leakIndex)
                actions += 1
                if not leak_found:
                    # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
                    MAY_CONTAIN_LEAK -= detection_range
    
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear()  
    # print("Size: ", graphSize, " -> Bot1 Total Actions: ", actions)
    return actions
    
def testBot2(k_value):
    graphSize = 30
    graph, n = initGraph(graphSize)
    botIndex = random.choice(OPEN_SQUARES)
    leakIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)
    leakIndex = placeLeak(graph, leakIndex, botIndex, n, False, k_value)
    MAY_CONTAIN_LEAK = set(OPEN_SQUARES)
    b2 = bot.Bot2(botIndex)
    actions = 0
    leak_found = False

    # Scan all the nodes in the detection radius
    detection_range, leak_found = b2.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK, leakIndex)
    actions += 1

    if not leak_found:
      # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
      MAY_CONTAIN_LEAK -= detection_range
      
    while not b2.succeeded:
        # If a leak is found, go through all the nodes in the detection range starting from the closest one
        if leak_found:
            closest_node, distance = b2.find_closest_node(graph, n, detection_range)
            actions += distance
            b2.step(graph, closest_node) # Move to the closest node in detection range
            if(b2.succeeded):
                break
            else:
                detection_range.remove(b2.currIndex) # If the space is not a leak, remove it
        else:
            # If a leak is not found, go to the closest unchecked node with the most unchecked nodes in its area
            best_node, distance = b2.find_closest_best_node(graph, n, k_value, MAY_CONTAIN_LEAK)
            actions += distance
            b2.step(graph, best_node)
            if b2.succeeded:
                break
            else: # Scan all the nodes in the detection radius if the bot hasn't landed on the leak position
                MAY_CONTAIN_LEAK.remove(b2.currIndex)
                detection_range, leak_found = b2.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK, leakIndex)
                actions += 1
                if not leak_found:
                    # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
                    MAY_CONTAIN_LEAK -= detection_range
    
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear()  
    # print("Size: ", graphSize, " -> Bot2 Total Actions: ", actions)
    return actions
    
def testBot3(a):
    ALPHA = a
    
    start = time.time()
    graph, n = initGraph(30)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex, botIndex, n, True, 0)

    printGraph(graph, n)
    
    
    b3 = bot.Bot3(botIndex)
    actions = 0
    MAY_CONTAIN_LEAK = OPEN_SQUARES.copy()
    MAY_CONTAIN_LEAK.remove(botIndex)
    PROBABILITY_MAP = initProbability(n, MAY_CONTAIN_LEAK)
    
    allDist = b3.initDistance(graph, n, OPEN_SQUARES)
    
    while not b3.succeeded:

        # Bot senses, check if there is a beep or not, update probabilities of cells accordingly
        beep, beepProb = b3.beep(ALPHA, graph, n, leakIndex, allDist)
        actions += 1
        if beep:
            PROBABILITY_MAP = b3.beepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)
        else:
            PROBABILITY_MAP = b3.noBeepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)

        # Find the node to move to that has the highest probability
        highestProbabilities = b3.getHighestProbabilities(PROBABILITY_MAP)
        best_node, distance = b3.find_closest_node(graph, n, highestProbabilities)
        path = b3.getShortestPath(graph, n, b3.currIndex, best_node)
        path.pop(0)
        for x in path:
            b3.step(graph, x)
            actions += 1
            if not b3.succeeded:
                if b3.currIndex in MAY_CONTAIN_LEAK:
                    MAY_CONTAIN_LEAK.remove(b3.currIndex)
                    PROBABILITY_MAP[b3.currIndex] = 0
                    b3.moveUpdate(PROBABILITY_MAP)
            else:
                break
        
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear()    
    
    printGraph(graph,n)
    print("done : " + str(actions) + " actions taken")
    end = time.time()
    
    print("Program finished in " + str(end-start) + " seconds")
    return actions

def testBot4(a):
    graphSize = 30
    graph, n = initGraph(graphSize)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)
    OPEN_SQUARES.remove(botIndex)
    leakIndex = random.choice(OPEN_SQUARES)
    leakIndex = placeLeak(graph, leakIndex, botIndex, n, True, 0)
    MAY_CONTAIN_LEAK = OPEN_SQUARES.copy()
    PROBABILITY_MAP = initProbability(n, MAY_CONTAIN_LEAK)
    b4 = bot.Bot4(botIndex)
    actions = 0

    while not b4.succeeded:

        # Bot senses, check if there is a beep or not, update probabilities of cells accordingly
        beep, beepProb, distance = b4.beep(a, graph, n, leakIndex)
        actions += 1
        if beep:
            PROBABILITY_MAP = b4.beepUpdate(PROBABILITY_MAP, graph, n, a)
        else:
            PROBABILITY_MAP = b4.noBeepUpdate(PROBABILITY_MAP, graph, n, a)

        # Find the nodes with highest probabilities
        highestProbabilities = b4.getHighestProbabilities(PROBABILITY_MAP)

        # If these nodes have the same distance, choose the node that has the most unchecked cells (which cells have the most open neighbor cells with prob > 0)
        best_node, distance = b4.find_closest_best_node(graph, n, highestProbabilities, PROBABILITY_MAP)
        path = b4.getShortestPath(graph, n, b4.currIndex, best_node)
        path.pop(0)
        for x in path: # As the bot travels to that designated node, for every space that isn't a leak, update that space with 0 probability and update the rest of the probabilities accordingly
            b4.step(graph, x)
            actions += 1
            if not b4.succeeded:
                if b4.currIndex in MAY_CONTAIN_LEAK:
                    MAY_CONTAIN_LEAK.remove(b4.currIndex)
                    PROBABILITY_MAP[b4.currIndex] = 0
                    b4.moveUpdate(PROBABILITY_MAP)
            else:
                break
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear()  
    # print("Bot4 Total Actions: ", actions)
    return actions

    
def testBot5(k_value):
    graphSize = 30
    graph, n = initGraph(graphSize)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex1 = random.choice(OPEN_SQUARES)
    leakIndex1 = placeLeak(graph, leakIndex1, botIndex, n, False, k_value)

    leakIndex2 = random.choice(OPEN_SQUARES)
    leakIndex2 = placeLeak(graph, leakIndex2, botIndex, n, False, k_value)

    MAY_CONTAIN_LEAK = set(OPEN_SQUARES)
    b5 = bot.Bot5(botIndex)
    actions = 0
    leak_found = False

    # Scan all the nodes in the detection radius
    detection_range, leak_found = b5.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
    actions += 1

    if not leak_found:
      # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
      MAY_CONTAIN_LEAK -= detection_range
    while not b5.succeeded2:
        # If a leak is found, go through all the nodes in the detection range starting from the closest one
        if leak_found:
            closest_node, distance = b5.find_closest_node(graph, n, detection_range)
            actions += distance
            b5.step(graph, closest_node) # Move to the closest node in detection range

            # If the first leak is found, remove it from MAY_CONTAIN_LEAK and set leak_found as false
            # Scan to see if there's another leak in the vicinity
            # Proceed to find the second leak as if you are trying to find the first leak
            if b5.succeeded and not b5.succeeded2: 
                leak_found = False
                MAY_CONTAIN_LEAK.remove(b5.currIndex)
                detection_range, leak_found = b5.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
                actions += 1
            elif b5.succeeded2:
                break
            else:
                # If the space is not a leak, remove it
                MAY_CONTAIN_LEAK.remove(b5.currIndex)
                detection_range.remove(b5.currIndex) 
        else:
          # If a leak is not found, go to the closest node in MAY_CONTAIN_LEAK
          closest_node, distance = b5.find_closest_node(graph, n, MAY_CONTAIN_LEAK)
          actions += distance
          b5.step(graph, closest_node) # Move to the closest node in MAY_CONTAIN_LEAK

          if b5.succeeded2:
              break
          else: # Scan all the nodes in the detection radius
              MAY_CONTAIN_LEAK.remove(b5.currIndex)
              detection_range, leak_found = b5.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
              actions += 1
              if not leak_found:
                  # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
                  MAY_CONTAIN_LEAK -= detection_range
    
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear() 
    # print("Bot5 Total Actions: ", actions)
    return actions

def testBot6(k_value):
    graph, n = initGraph(30)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex1 = random.choice(OPEN_SQUARES)
    graph[leakIndex1] = LEAK
    leakIndex1 = placeLeak(graph, leakIndex1, botIndex, n, False, k_value)

    leakIndex2 = random.choice(OPEN_SQUARES)
    graph[leakIndex2] = LEAK
    leakIndex2 = placeLeak(graph, leakIndex2, botIndex, n, False, k_value)

    MAY_CONTAIN_LEAK = set(OPEN_SQUARES)
    b6 = bot.Bot6(botIndex)
    actions = 0
    leak_found = False

    # Scan all the nodes in the detection radius
    detection_range, leak_found = b6.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
    actions += 1

    if not leak_found:
      # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
      MAY_CONTAIN_LEAK -= detection_range

    while not b6.succeeded2:
        # If a leak is found, go through all the nodes in the detection range starting from the closest one
        if leak_found:
            closest_node, distance = b6.find_closest_node(graph, n, detection_range)
            actions += distance
            b6.step(graph, closest_node) # Move to the closest node in detection range

            # If the first leak is found, remove it from MAY_CONTAIN_LEAK and set leak_found as false
            # Scan to see if there's another leak in the vicinity
            # Proceed to find the second leak as if you are trying to find the first leak
            if b6.succeeded and not b6.succeeded2:
                leak_found = False
                MAY_CONTAIN_LEAK.remove(b6.currIndex)
                detection_range, leak_found = b6.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
                actions += 1
            elif b6.succeeded2:
                break
            else:
                # If the space is not a leak, remove it
                MAY_CONTAIN_LEAK.remove(b6.currIndex)
                detection_range.remove(b6.currIndex) 
        else:
          # If a leak is not found, go to the closest unchecked node with the most unchecked nodes in its area
          closest_node, distance = b6.find_closest_best_node(graph, n, k_value, MAY_CONTAIN_LEAK)
          actions += distance
          b6.step(graph, closest_node) 

          if b6.succeeded2:
              break
          else: # Scan all the nodes in the detection radius
              MAY_CONTAIN_LEAK.remove(b6.currIndex)
              detection_range, leak_found = b6.check_detection_radius(graph, n, k_value, MAY_CONTAIN_LEAK)
              actions += 1
              if not leak_found:
                  # Remove all the nodes in the detection radius that are in MAY_CONTAIN_LEAK
                  MAY_CONTAIN_LEAK -= detection_range
    
    OPEN_SQUARES.clear()
    PROBABILITY_MAP.clear()
    MAY_CONTAIN_LEAK.clear() 
    # print("Bot6 Total Actions: ", actions)
    return actions

def testBot7(a):
    
    graph, n = initGraph(30)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex1 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex1, botIndex, n, True, 0)

    leakIndex2 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex2, botIndex, n, True, 0)

    b7 = bot.Bot7(botIndex)
    actions = 0
    heatMap1 = b7.initProbability(graph, n)
    heatMap2 = b7.initProbability(graph, n)
    allDist = b7.initDistance(graph, n, OPEN_SQUARES)
    
    while not b7.succeeded:
        
        b7.botEntersCellUpdate(heatMap1, graph, n)
        beeped, prob = b7.beep(a, graph, n, leakIndex1, allDist)
        
        if beeped:
            b7.beepUpdateProb(heatMap1, prob, graph, n,a, allDist)
        else:
            b7.noBeepUpdateProb(heatMap1, prob, graph, n,a, allDist)        
        
        path = b7.replan(graph, n, heatMap1)
        if path == None:
            continue
        
        nextStep = path.pop(0)
        
        if nextStep == b7.currIndex:
            if len(path) == 0:
                continue
            
            nextStep = path.pop(0)
        
        b7.step(graph, nextStep)
        
        actions += 1

#        if actions == 5000:
#            print("BOT FAILED")
#            OPEN_SQUARES.clear()
#            return(5000)
        
    printGraph(graph, n)        
    
    openSquare(graph, leakIndex1)
    
    while not b7.succeeded2:
        
        b7.botEntersCellUpdate(heatMap2, graph, n)
        beeped, prob = b7.beep(a, graph, n, leakIndex2, allDist)
        
        if beeped:
            b7.beepUpdateProb(heatMap2, prob, graph, n,a, allDist)
        else:
            b7.noBeepUpdateProb(heatMap2, prob, graph, n,a, allDist)        
        
        path = b7.replan(graph, n, heatMap2)
        if path == None:
            continue
        
        nextStep = path.pop(0)
        
        if nextStep == b7.currIndex:
            if len(path) == 0:
                continue
            
            nextStep = path.pop(0)
        
        b7.step(graph, nextStep)
        
        actions += 1
        
        if b7.currIndex == leakIndex2:
            b7.succeeded2()
            

    printGraph(graph, n)        
    print("done : " + str(actions) + " actions taken")
    
    OPEN_SQUARES.clear()
    return actions
    
    
    
def testBot8(a):

    ALPHA = a
    start = time.time()
    
    graph, n = initGraph(30)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex1 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex1, botIndex, n, True, 0)

    leakIndex2 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex2, botIndex, n, True, 0)

    b8 = bot.Bot8(botIndex)
    actions = 0
    heatMap = b8.initProbability(graph, n)
    allDist = b8.initDistance(graph, n, OPEN_SQUARES)
    
    printGraph(graph, n)
    while not b8.succeeded:
        
        b8.botEntersCellUpdate(heatMap, graph, n)
        beeped, prob, p1, p2 = b8.beep2leaks(a, graph, n, leakIndex1, leakIndex2, allDist)
        
        if beeped:
            b8.beepUpdateProb(heatMap, prob, graph, n,a, allDist)
        else:
            b8.noBeepUpdateProb(heatMap, prob, graph, n,a, allDist)        
        
        path = b8.replan(graph, n, heatMap)
        if path == None:
            continue
        
        nextStep = path.pop(0)
        
        if nextStep == b8.currIndex:
            if len(path) == 0:
                continue
            
            nextStep = path.pop(0)
        
        b8.step(graph, nextStep)
        
        actions += 1
        if actions % 100 == 0:
            print("Current Actions Taken : " + str(actions))
            printGraph(graph, n)
        
#        if actions == 5000:
#            print("BOT FAILED")
#            OPEN_SQUARES.clear()
#            return(5000)
         
    printGraph(graph, n)        
    nextTarget = 0
    
    if b8.currIndex == leakIndex1:
        graph[leakIndex1] = OPEN
        nextTarget = leakIndex2
    else:
        graph[leakIndex2] = OPEN
        nextTarget = leakIndex1
    
    MAY_CONTAIN_LEAK = OPEN_SQUARES.copy()
    MAY_CONTAIN_LEAK.remove(b8.currIndex)
    PROBABILITY_MAP = initProbability(n, MAY_CONTAIN_LEAK)
    b8.succeeded = False
    
    while b8.currIndex != nextTarget:

        # Bot senses, check if there is a beep or not, update probabilities of cells accordingly
        beep, beepProb = b8.beep(ALPHA, graph, n, nextTarget, allDist)
        actions += 1
        if beep:
            PROBABILITY_MAP = b8.beepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)
        else:
            PROBABILITY_MAP = b8.noBeepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)

        # Find the node to move to that has the highest probability
        highestProbabilities = b8.getHighestProbabilities(PROBABILITY_MAP)
        best_node, distance = b8.find_closest_node(graph, n, highestProbabilities)
        path = b8.getShortestPath(graph, n, b8.currIndex, best_node)
        path.pop(0)
        for x in path:
            b8.step(graph, x)
            actions += 1
            
            if not b8.succeeded:
                if b8.currIndex in MAY_CONTAIN_LEAK:
                    MAY_CONTAIN_LEAK.remove(b8.currIndex)
                    PROBABILITY_MAP[b8.currIndex] = 0
                    b8.moveUpdate(PROBABILITY_MAP)
            else:
                break
    
    printGraph(graph,n)
    print("done : " + str(actions) + " actions taken")
    end = time.time()
    
    print("Program finished in " + str(end-start) + " seconds")
    
    MAY_CONTAIN_LEAK.clear()
    PROBABILITY_MAP.clear()
    OPEN_SQUARES.clear()
    return actions
    
def testBot9(a):
    ALPHA = a
    start = time.time()
    
    graph, n = initGraph(30)
    botIndex = random.choice(OPEN_SQUARES)
    placeBot(graph, botIndex)

    leakIndex1 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex1, botIndex, n, True, 0)

    leakIndex2 = random.choice(OPEN_SQUARES)
    placeLeak(graph, leakIndex2, botIndex, n, True, 0)

    b9 = bot.Bot9(botIndex)
    actions = 0
    heatMap = b9.initProbability(graph, n)
    allDist = b9.initDistance(graph, n, OPEN_SQUARES)
    
    printGraph(graph, n)
    while not b9.succeeded:
        
        b9.botEntersCellUpdate(heatMap, graph, n)
        
        for i in range(5):
            beeped, prob, p1, p2 = b9.beep2leaks(a, graph, n, leakIndex1, leakIndex2, allDist)
            if beeped:
                b9.beepUpdateProb(heatMap, prob, graph, n,a, allDist)
            else:
                b9.noBeepUpdateProb(heatMap, prob, graph, n,a, allDist)        
        
        path = b9.replan(graph, n, heatMap)
        if path == None:
            continue
        
        nextStep = path.pop(0)
        
        if nextStep == b9.currIndex:
            if len(path) == 0:
                continue
            
            nextStep = path.pop(0)
        
        b9.step(graph, nextStep)
        
        actions += 1
        if actions % 100 == 0:
            print("Current Actions Taken : " + str(actions))
            printGraph(graph, n)
#        
#        if actions == 5000:
#            print("BOT FAILED")
#            OPEN_SQUARES.clear()
#            return(10000)
         
    printGraph(graph, n)        
    nextTarget = 0
    
    if b9.currIndex == leakIndex1:
        graph[leakIndex1] = OPEN
        nextTarget = leakIndex2
    else:
        graph[leakIndex2] = OPEN
        nextTarget = leakIndex1
    
    MAY_CONTAIN_LEAK = OPEN_SQUARES.copy()
    MAY_CONTAIN_LEAK.remove(b9.currIndex)
    PROBABILITY_MAP = initProbability(n, MAY_CONTAIN_LEAK)
    b9.succeeded = False
    
    while b9.currIndex != nextTarget:

        # Bot senses, check if there is a beep or not, update probabilities of cells accordingly
        beep, beepProb = b9.beep(ALPHA, graph, n, nextTarget, allDist)
        actions += 1
        if beep:
            PROBABILITY_MAP = b9.beepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)
        else:
            PROBABILITY_MAP = b9.noBeepUpdateProb(PROBABILITY_MAP, beepProb, graph, n, ALPHA, allDist)

        # Find the node to move to that has the highest probability
        highestProbabilities = b9.getHighestProbabilities(PROBABILITY_MAP)
        best_node, distance = b9.find_closest_node(graph, n, highestProbabilities)
        path = b9.getShortestPath(graph, n, b9.currIndex, best_node)
        path.pop(0)
        for x in path:
            b9.step(graph, x)
            actions += 1
            
            if not b9.succeeded:
                if b9.currIndex in MAY_CONTAIN_LEAK:
                    MAY_CONTAIN_LEAK.remove(b9.currIndex)
                    PROBABILITY_MAP[b9.currIndex] = 0
                    b9.moveUpdate(PROBABILITY_MAP)
            else:
                break
    
    printGraph(graph,n)
    print("done : " + str(actions) + " actions taken")
    end = time.time()
    
    print("Program finished in " + str(end-start) + " seconds")
    
    MAY_CONTAIN_LEAK.clear()
    PROBABILITY_MAP.clear()
    OPEN_SQUARES.clear()
    return actions
    
def main():
    testBot9(1)


    
if __name__ == "__main__":
    main()