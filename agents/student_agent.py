# Student agent: Add your own agent here 
from queue import PriorityQueue
from agents.agent import Agent
from store import register_agent
import numpy as np
from typing import Tuple, List, Set, Generator
from xmlrpc.client import Boolean
from itertools import combinations_with_replacement, count
import time
from math import sqrt, log

#TODO: check RAM

class Move:
    position : Tuple[int]
    barrier : List[int]

class State: #FIXME : when turn is true: add, when false, subtract
    def __init__(self, chess_board:np.ndarray, my_pos:Tuple[int,int], adv_pos:Tuple[int,int], turn:Boolean):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.turn = turn

    def __hash__(self):
        return hash((str(self.chess_board), self.my_pos, self.adv_pos, self.turn))

    def __eq__(self, other):
        if type(other) is type(self):
            return (np.all(self.chess_board == other.chess_board)) and (self.my_pos == other.my_pos) and (self.adv_pos == other.adv_pos)
        else:
            return False
    
    def is_endgame(self):
            # Union-Find
            father = dict()
            for r in range(self.chess_board.shape[0]):
                for c in range(self.chess_board.shape[1]):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            for r in range(self.chess_board.shape[0]):
                for c in range(self.chess_board.shape[1]):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if self.chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(self.chess_board.shape[0]):
                for c in range(self.chess_board.shape[1]):
                    find((r, c))
            p0_r = find(self.my_pos) #TODO: change 
            p1_r = find(self.adv_pos)
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r: #if same parent -> same region
                return None
            # else not same parent -> different regions
            player_win = None #winner position
            if p0_score > p1_score:
                player_win = self.my_pos
            elif p0_score < p1_score:
                player_win = self.adv_pos
            elif p0_score == p1_score:
                player_win = "TIE"
            
            #print("my pos in is_end_game: ", self.my_pos)

            return player_win

    def get_legal_moves(self, max_step:int) -> Set: 
        out = set()

        # generate valid moves [,]
        for k in range(1,max_step+1):
            # get random combination
            poss_paths = combinations_with_replacement(["U","D","L","R"], k)
            #poss_paths = random.shuffle(poss_paths)

            for path in poss_paths:
                if self.turn:
                    old_pos = self.my_pos
                else:
                    old_pos = self.adv_pos

                end_pos = old_pos # var declaration

                for dir in path:
                    if dir == "U": #x-1
                        new_pos = (end_pos[0]-1,end_pos[1])
                        end_pos = new_pos

                    elif dir == "D": #x+1
                        new_pos = (end_pos[0]+1,end_pos[1])
                        end_pos = new_pos  
                    
                    elif dir == "L": #y-1
                        new_pos = (end_pos[0],end_pos[1]-1)
                        end_pos = new_pos
                    
                    elif dir == "R": #y+1
                        new_pos = (end_pos[0],end_pos[1]+1)
                        end_pos = new_pos
                
                # put barriers
                for dir in range(4):
                    if (end_pos,dir) not in out:
                        r, c = end_pos
                        if 0 <= r < len(self.chess_board) and 0 <= c < len(self.chess_board): # check boundary
                            if self.check_valid_step(old_pos, end_pos, dir, max_step): #check valid step
                                out.add((end_pos,dir))
        return out

    # TODO: check if good
    def check_valid_step(self, start_pos:Tuple, end_pos:np.ndarray, barrier_dir:int, max_step:int) -> Boolean:
        """
        Check if the step the agent takes is valid (reachable and within max steps).
        """

        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = self.my_pos if not self.turn else self.adv_pos #TODO: change not if needed

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos

            if cur_step == max_step:
                break
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            for dir, move in enumerate(moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def neighbors(self, max_step:int) -> List: 
        my_moves = self.get_legal_moves(max_step)
        move_state_pairs = []
        for move in my_moves:
            new_state = self.make_move(move)
            move_state_pairs.append((new_state, move))
        return move_state_pairs
            
    def make_move(self, move:Tuple):
        new_chess_board = self.chess_board.copy()
        (new_pos, new_barr) = move 
        new_chess_board[new_pos][new_barr] = True
        if self.turn:
            return State(new_chess_board, new_pos, self.adv_pos, False)
        else:
            return State(new_chess_board, self.my_pos, new_pos, True)
        
    # TODO: add controlled region, block opponent, cells behind
    def evaluation(self) -> float: #weighted sum of all heuristics
        return self.win_next()
        # return self.break_through(dir) + self.connected_barrier(dir) + self.controlled_center()
        # self.win_next() + 
    
    #0
    def win_next(self) -> float: #FIXME: 78% win rate
        player_win = self.is_endgame()
        if player_win == self.my_pos:
            #print("ME WINNER!!!: ", player_win)
            #print(" -- my pos: ", self.my_pos)
            return 1000

        elif player_win == self.adv_pos:
            #print("ADV WINNER!!!: ", player_win)
            #print(" -- adv pos: ", self.adv_pos)
            return -1000

        # if tie or not end game
        return 0
        
    #1
    def cells_behind(self, chess_board, my_pos, adv_pos, max_step)-> float:
        return 0

    # helper for 2 and 3: return int (0: no side connected, 1: one side connected, 2: both sides connected)
    def is_barrier_connected(self, dir): # for udlr in pos, for i in range(3)
        if self.turn:
            pos = self.my_pos
        else:
            pos = self.adv_pos
        numside1 = 0 # max 6 connected barriers
        numside2 = 0 
        x = pos[0]
        y = pos[1]
        # horizontal barrier
        if dir in [0,2]: 
            if self.chess_board[x,y,1]: # check this @ barrier right
                numside1+=1
            if self.chess_board[x,y,3]: # check this @ barrier left
                numside2+=1

            try:
                if self.chess_board[x,y-1,dir]: # check adjacent left @ barrier north or south
                    numside2+=1
            except:
                pass

            try:
                if self.chess_board[x,y+1,dir]: # check adjacent right @ barrier north or south
                    numside1+=1
            except:
                pass
                
            if dir == 0: # @ barrier north
                newx = x-1 # go up

            else: # == 2 @ barrier south
                newx = x+1 # go down

            try:
                if self.chess_board[newx,y,1]: #check opposite @ barrier right
                    numside1+=1
            except:
                pass

            try:
                if self.chess_board[newx,y,3]:  #check opposite @ barrier left
                    numside2+=1
            except:
                pass
            
        # vertical barrier
        elif dir in [1,3]: 
            if self.chess_board[x,y,0]: # check this @ barrier north
                numside1+=1
            if self.chess_board[x,y,2]: # check this @ barrier south
                numside2+=1

            try:
                if self.chess_board[x-1,y,dir]: # check adjacent up @ barrier left or right
                    numside1+=1
            except: 
                pass
            try:
                if self.chess_board[x+1,y,dir]: # check adjacent down @ barrier left or right
                    numside2+=1
            except: 
                pass

            if dir == 1: # @ barrier right
                newx = y+1 # go right

            else: # dir == 3 @ barrier left
                newy = y-1 # go left

            try:
                if self.chess_board[x,newy,0]: #check other @ barrier north south
                    numside1+=1
            except: 
                pass

            try:
                if self.chess_board[x,newy,2]:
                    numside2+=1
            except:
                pass
        
        return [numside1, numside2]

    #2 stay next to a unconnected barrier (1 or 2 sides end-point) 
    #NOTE: important for mid game
    def break_through(self, dir) -> float:
        stats = self.is_barrier_connected(dir)
        total = 2
        if stats[0] > 0: # side 1 connected
            total -= 1
        
        if stats[1] > 0:  # side 2 connected
            total -= 1

        if not self.turn:
            return 0-total
        
        return float(total)

    #3 connect the barrier that you put with an existing barrier
    #NOTE: important for start game
    def connected_barrier(self, dir) -> float:
        stats = self.is_barrier_connected(dir)
        
        if sum(stats) >= 1 and self.turn:
            return float(1)
        elif sum(stats) >= 1 and not self.turn:
            return float(-1)
        else:
            return float(0) # not connected at all
            
    #4 put barrier around OR squeeze him in corner
    # early game plan
    # def block_opponent(self, dir, my_pos, adv_pos):
    #     if self.turn:
    #         pos = self.my_pos
    #     else:
    #         pos = self.adv_pos
        
    #     if self.chess_board[x,y,1]: # check this @ barrier right

    #5 return (0: on the side, +1 for each layer closer to the middle)
    #NOTE: important in early game since by going near the center, we can control a bigger region ->
    # assign a smaller score as it will no longer be useful in late game (score: 0 to half of board*2)
    def controlled_center(self) -> float: 
        middle = len(self.chess_board)//2
        if self.turn:
            pos = self.my_pos
        else:
            pos = self.adv_pos
        score = 0
        x = pos[0]
        y = pos[1]
        if len(self.chess_board) % 2 == 0: #even board size
            middle1 = middle-1 # 2 middles
            if x<=middle1: # left side of middle
                score+=x
            elif x>=middle: # right side of middle
                score+=middle-(x-middle+1)

            if y<=middle1: # above middle
                score+=y
            elif y>=middle: # below middle
                score+=middle-(y-middle+1) 

        else: #odd board size
            if x<=middle: # left side of middle
                score+=x
            else: # right side of middle
                score+= middle-(x-middle) 

            if y<=middle: # above middle
                score+=y 
            else: # below middle
                score+= middle-(y-middle) 

        if not self.turn:
            return float(0-score)
        return float(score)

    #6
    def controlled_region(self, chess_board, my_pos, adv_pos, max_step):
        # min_num_barr = 0
        # # check if it is possible to close a region with at most 3 additional barriers 
        # # while maximizing the region controlled by player
        # for i in range(3):
            
        
        return 0

    # 6.1
    def define_region(self, chess_board, my_pos, adv_pos, max_step):
        return 0

class MTCNode:
    state : State
    score : float
    children : List["MTCNode"] 
    parent : "MTCNode"
    move : Tuple[Tuple[int, int], int]
    def __init__(self, state:State, parent:"MTCNode", move:Tuple[Tuple[int, int], int]):
        self.state = state
        self.score = state.evaluation() #state.evaluation(self.move[1]) #heuristic\ self.move[1] is barrier dir
        self.children = []
        self.parent = parent
        self.move = move

    def add(self, node:"MTCNode") -> None:
        self.children.append(node)
        self.score
    
    #NOTE: include UCT heuristic if possible
    def UCT(self, all_child): # heuristic for next move to choose
        allUCT = []
        for child in all_child:
            CONSTANT = 1 #TODO: change 
            UCTscore = child.wins/child.visits + CONSTANT * sqrt(2*log(self.visits)/child.visits)
            allUCT.append(UCTscore)
        sorted(allUCT)
        return allUCT[-1]

    def select(self, max_depth, visited:Set) -> Generator[Tuple["MTCNode",float], None, None]: # lazytraversal that selects a leaf at max_depth
        #print("self (MTCNode) returned from select ",self)
        #print("self.move inside select ",self.move)
        #print("self.state inside select ",self.state)
        if max_depth == 0:
            yield (self, max_depth)
            return

        visited.add(self)

        if not self.children:
            yield (self, max_depth)# ,max_depth:  if max depth != 0, can continue to decrement in expand function
            return

        for child in self.children:
            if child not in visited:
                yield from child.select(max_depth-1, visited)
        return

    def aggregate(self, scores):
        return max(scores, default=self.score)


    #TODO: expand should be good
    def expand(self, max_step:int, depth:int) -> Generator[float, None, None]:
        #print("leaf(MTCNode) in expand ",self)
        #print("leaf.move in expand ",self.move)
        # print("self.state in expand ",self.state)

        queue = PriorityQueue()

        probe = count()

        queue.put((-self.score, depth, next(probe), self))

        while not queue.empty():
            score, depth, _, node = queue.get()
            yield score

            if depth == 0:
                return

            for move in node.state.get_legal_moves(max_step):#TODO: randomly sample neighboring states [(state1,move1),(state2,move2)]
                child = MTCNode(node.state.make_move(move), node, move)
                node.children.append(child)
                node.backpropagate()

                queue.put((-child.score, depth - 1, next(probe), child))

        return


    def backpropagate(self):
        self.score = self.aggregate((child.score for child in self.children))
        if self.parent:
            self.parent.backpropagate()


    def run(self, max_step, max_depth):
        start_time = time.monotonic() # return curr time from monotonic clock
        visited = set()

        # expansion, simulation and backpropagation for each leaf
        for leaf, depth in self.select(max_depth, visited): #NOTE: leaves is type generator
            for _ in leaf.expand(max_step, depth):
                if time.monotonic() - start_time > 2:
                    return

    def get_best_move(self, max_step, max_depth) -> Tuple["MTCNode",float]:
        self.run(max_step, max_depth)
        best_node = max(self.children, key=lambda child: child.score)
        # return the (node,score) combination with the highest score
        #out = max(scores,key=lambda item:item[1])
        #print("best.move in gbm ", best_node.move) #BUG: None
        return best_node

    def find(self, state:State, max_depth:int) -> bool:
        if self.state == state:
            return self
        if max_depth == 0:
            return None
        for child in self.children:
            return child.find(state, max_depth-1)
        return None



@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        self.mtcTree = None
    
        # Returns optimal value for current player
    #(Initially called for root and maximizer)
    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        state = State(chess_board,my_pos,adv_pos,True) #How to check if our turn is True

        if self.mtcTree is None:
            self.mtcTree = MTCNode(state,None,None)
        else:
            self.mtcTree = self.mtcTree.find(state, 1)
            if self.mtcTree is None:
                self.mtcTree = MTCNode(state, None, None)
            else:
                self.mtcTree.parent = None

        node = self.mtcTree.get_best_move(max_step, max_depth=10)

        return node.move
 # do this only once        (node n ode.get_best_move(max_step) #returns (best_node,max_score)

        #return node.move
        # implement search <- call heuristic function <- call other necessary functions
        # use minmax -> does not affect utility against random agent
        # pruning (alpha-beta or prune the obviously bad moves with heuristic)
        # generate all legal moves -> good for bfs cus it generates one whole layer
                
                
        # state = tree node
        # go down tree, and find all leaves at max_depth
        # then get evaluation score for each of those leaves with state.evaluate
        # aggregate evaluation score of all the leaves max/average/... (to get the score for going to that state)
        # choose the best one
        # recurse till the game is ended

        # keys  dans dict = get_legal_moves: {move1: score, move2: score,...}
        # not interested in move 1.1 move 1.1.1 move 1.1.1.1(l)a
    

