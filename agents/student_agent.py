# Student agent: Add your own agent here 
from queue import PriorityQueue
from agents.agent import Agent
from store import register_agent
import numpy as np
from typing import Tuple, List, Set, Generator
from xmlrpc.client import Boolean
from itertools import combinations_with_replacement, count
import time

class Move:
    position : Tuple[int]
    barrier : List[int]

class State:
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
            p0_r = find(self.my_pos)
            p1_r = find(self.adv_pos)
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r: #if same parent -> same region
                return None
            # else not same parent -> different regions
            player_win = None #winner
            if p0_score > p1_score:
                player_win = self.my_pos
            elif p0_score < p1_score:
                player_win = self.adv_pos
            elif p0_score == p1_score:
                player_win = "TIE"
            return player_win

    def get_legal_moves(self, max_step:int) -> Set: 
        out = set()

        # generate valid moves
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

    # Get neighboring states by applying make_move()
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
        
    def evaluation(self) -> float: #weighted sum of all heuristics
        return self.win_next()
    
    # Check if state is a winning state
    def win_next(self) -> float: 
        player_win = self.is_endgame()
        if player_win == self.my_pos:
            return 1000

        elif player_win == self.adv_pos:
            return -1000

        return 0

class MCTNode:
    state : State
    score : float
    children : List["MCTNode"] 
    parent : "MCTNode"
    move : Tuple[Tuple[int, int], int]
    def __init__(self, state:State, parent:"MCTNode", move:Tuple[Tuple[int, int], int]):
        self.state = state
        self.score = state.evaluation()
        self.children = []
        self.parent = parent
        self.move = move

    def add(self, node:"MCTNode") -> None:
        self.children.append(node)
        self.score

    # lazytraversal that selects a leaf at max_depth
    def select(self, max_depth, visited:Set) -> Generator[Tuple["MCTNode",float], None, None]: 
        if max_depth == 0:
            yield (self, max_depth)
            return

        visited.add(self)

        if not self.children:
            yield (self, max_depth)
            return

        for child in self.children:
            if child not in visited:
                yield from child.select(max_depth-1, visited)
        return

    # aggregation of heuristic scores by taking the maximum
    def aggregate(self, scores):
        return max(scores, default=self.score)

    # expansion step of MCT
    def expand(self, max_step:int, depth:int) -> Generator[float, None, None]:
        queue = PriorityQueue()
        probe = count()
        queue.put((-self.score, depth, next(probe), self))

        while not queue.empty():
            score, depth, _, node = queue.get()
            yield score

            if depth == 0:
                return

            for move in node.state.get_legal_moves(max_step):
                child = MCTNode(node.state.make_move(move), node, move)
                node.children.append(child)
                node.backpropagate()
                queue.put((-child.score, depth - 1, next(probe), child))
        return

    # backprogation step of MCT
    def backpropagate(self):
        self.score = self.aggregate((child.score for child in self.children))
        if self.parent:
            self.parent.backpropagate()

    # search
    def run(self, max_step, max_depth):
        start_time = time.monotonic() # return curr time from monotonic clock
        visited = set()

        # expansion, simulation and backpropagation for each leaf
        for leaf, depth in self.select(max_depth, visited):
            for _ in leaf.expand(max_step, depth):
                if time.monotonic() - start_time > 2:
                    return

    # get node with the highest heuristic score
    def get_best_move(self, max_step, max_depth) -> Tuple["MCTNode",float]:
        self.run(max_step, max_depth)
        best_node = max(self.children, key=lambda child: child.score)
        return best_node

    # find child
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
        self.mctTree = None
    
    # Returns optimal value for current player
    #(Initially called for root and maximizer)
    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        
        state = State(chess_board,my_pos,adv_pos,True)

        if self.mctTree is None:
            self.mctTree = MCTNode(state,None,None)
        else:
            self.mctTree = self.mctTree.find(state, 1)
            if self.mctTree is None:
                self.mctTree = MCTNode(state, None, None)
            else:
                self.mctTree.parent = None

        node = self.mctTree.get_best_move(max_step, max_depth=10)

        return node.move