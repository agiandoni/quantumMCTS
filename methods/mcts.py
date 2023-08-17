# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:00:44 2022

@author: Andoni Agirre
"""

import numpy as np
import math
import copy
from timeit import default_timer as timer

def get_game_value( board_state, player ):
    winner = board_state.check_winner()
    assert(winner != None)
    return winner*player


class Node():
    def __init__(self, board_state, parent=None, minimum_reward = -np.inf):
        self.visits = 0 #counts how many times the node has been visited during a simulation
        self.score = 0.0 #keeps the total score of the node
        self.fully_expanded = False #Initializes node as a leaf node
        self.board_state = board_state #This will store the state of our game; ie where we are on the game tree. It comes from some other code
        self.children = [] #The nodes to which we can go from the current node will be stored here; unexpanded for now (list of Nodes)
        self.parent = parent #The parent of the current node (this takes a Node() class object)
        #self.possible_moves = None

        self.possible_moves = self.board_state.moves_in_position()
        #Shuffle the moves here?
        
        ##NEW SINGLE-PLAYER ENHANCEMENTS
        #[BEST REWARD, LIST OF MOVES THAT LEAD THERE FROM EXPANDED NODE?]
        self.maximum_reward = minimum_reward #The score of the best rollout in the entire turn
        self.maximum_path = [] #Path from root to the best scoring node found in the entire turn
        #We start with the worst possible reward and update it with improvements from random rollouts
        
  # --- UCB SCORE ---
    def ucb1(self):
        if self.visits == 0:
            return math.inf
        self.ucb1_score = self.score/self.visits
        return self.score/self.visits #Rewards exploitation of well-scoring paths
    def ucb2(self):
        if self.visits == 0:
            return math.inf
        self.ucb2_score = np.sqrt(np.log(self.parent.visits)/self.visits)
        return np.sqrt(np.log(self.parent.visits)/self.visits) #Rewards exploration of nodes that were not visited a often


    # --- Selection stage methods ---


    def choose_favorite_child(self, adventurousness):
        #adventurousness is the C parameter in the UCB formula,
        #higher adventurousness will make the algorithm
        #venture into unexplored paths more often!

        assert(len(self.children) != 0)


        best_child_index = np.argmax([(child.score / child.visits) + adventurousness * np.sqrt(np.log(self.visits) / child.visits) for child in self.children])
        best_child = self.children[best_child_index]
        
        assert(best_child == max(self.children, key=lambda child: child.ucb1()+adventurousness*child.ucb2()))
        
        return max(self.children, key=lambda child: child.ucb1()+adventurousness*child.ucb2())
    
    def choose_particular_child(self, move):
        #Chooses the child with the state given by move.
        #Useful for memorizing good paths in the modified MCTS version
        assert(len(self.children) > 0), "choose_particular_child() called on childless node"
        Chosen_child = None
        for child in self.children:
            if np.isclose(child.board_state.state[-1], move):
                Chosen_child = child
        
        assert(Chosen_child is not None), "Wanted child was not found by choose_particular_child()"
        return Chosen_child
                

   
   # --- Expansion ---

    def add_child(self, child):
        self.children.append(child)


    def expand(self):

        #Stage 2 of MCTS - modified - it now expands them one by one, and actually travels to it before rollout
        move = self.possible_moves.pop() #Take only one of the unexpanded (Does not really matter which one we take, they're all ucb inf)

        new_board = copy.deepcopy(self.board_state)
        new_board.make_move(move)

        child = Node(board_state=new_board, parent=self) #Initialize the new node
        self.add_child(child)

        # Also need to check whether all the children have been expanded, and only then make self.expanded True
        if len(self.possible_moves) == 0:
            self.fully_expanded = True

        return child #Travels to the newly expanded child, ready for rolloout

    
    def expand_given_move(self, move):
        #Needed for modified MCTS
        
        #self.possible_moves.remove(move) #Comparing floats is complicated...
        for possibility in self.possible_moves:
            if np.isclose(possibility, move):
                self.possible_moves.remove(possibility)
        new_board = copy.deepcopy(self.board_state)
        new_board.make_move(move)

        child = Node(board_state=new_board, parent=self) #Initialize the new node
        self.add_child(child)

        # Also need to check whether all the children have been expanded, and only then make self.expanded True
        if len(self.possible_moves) == 0:
            self.fully_expanded = True

        return child #Travels to the newly expanded child, ready for rolloout
    
   # --- Rollout ---

    def random_step(self):
        moves = self.board_state.moves_in_position()
        move = np.random.randint(len(moves))
        return moves[move]

    def rollout_policy(self, moves):
        move = np.random.randint(len(moves))
        return moves[move]


    def rollout(self, player): #,position): #position is a Board object
        current_state = copy.deepcopy(self.board_state)
        moves_list = []
        while not current_state.is_it_over():
            legal_moves = current_state.moves_in_position()
            move = self.rollout_policy(legal_moves)
            moves_list.append(move)
            current_state = current_state.make_move(move)
            
            # SINGLE PLAYER OPTIMIZATION
            # self.maximum_path.append(move)
        #current_state.draw_board()
        # print(f"Current gameover position = {current_state.state}")
        # print(f"Moves_list in rollout = {moves_list}, state = {self.board_state.state}; returning {self.board_state.state+moves_list}")
        return get_game_value(current_state, player), self.board_state.state+moves_list


   # --- Backpropagation ---

    def backpropagation(self, end_score, alternating_rewards = False, who_played_last=None, moves_list=[]):
       
        if alternating_rewards: 
            #For 2 player games. get_game_value() already fixed the first end_score relative to the MCTS player, then we just alternate signs
            self.visits += 1
            self.score += (end_score*who_played_last) #We do -turn because -player played to get to the node where its player to play
            if self.parent:
                self.parent.backpropagation(end_score, True, -who_played_last)
        
        else:
            self.visits += 1
            self.score += end_score
            
            #NEW SINGLE PLAYER REWARDS (ONLY IF REWARDS DO NOT ALTERNATE)
            #print(f"UPDATING REWARD: {end_score}, {moves_list}")
            self.update_maximum_reward(end_score, moves_list)
            
            if self.parent:
                self.parent.backpropagation(end_score, moves_list=moves_list)
            
            
            

##################


    def travel(self, adventurousness = math.sqrt(2)): #Modified so as to use the new expansion method

        current_node = self

        while not current_node.board_state.is_it_over():
            if current_node.fully_expanded:
                current_node = current_node.choose_favorite_child(adventurousness)
            else:
                current_node = current_node.expand()

                #This loop should only terminate after a new node is expanded or a game over state is reach before expanding
                return current_node

        return current_node #This will execute if we reach a game over state during selection

    
    def MCTS_step(self, adventurousness: float, repeats: int, player: int, alternating_rewards= False, progress_intervals = 10, last_choice="max"):
        #Chooses the move to play
        start=timer()
        
        if progress_intervals:
            print(f"Simulation/{repeats}: ",end="")
        for simulation in range(repeats):
            
             if progress_intervals:
                 if simulation+1==repeats:
                     print(f"{simulation+1}.")
                 elif (simulation+1) % (repeats/progress_intervals) == 0:
                     print(f"{simulation+1}, ", end="")
        
            # Select until a leaf node is reached and expand
            destination = self.travel(adventurousness)
            destination.board_state.fix_turn() #unnecessary

            # Rollout until a game over node is reached
            end_score, moves_list = destination.rollout(player)
            
            #"Reward engineering" can safely happen below:
            #e.g. end_score = f(end_score, args)
            
            # Travel back up the tree and update nodes
            destination.backpropagation(end_score, alternating_rewards, -destination.board_state.turn*player, moves_list=moves_list)
        
        
        
        
        if last_choice == "ucb":
            return self.choose_favorite_child(adventurousness)
        elif last_choice == "max":
            return self.choose_favorite_child(0.0)
        elif last_choice == "robust":
            return max(self.children, key = lambda child: child.visits)
        elif last_choice == "compare":
            return [self.choose_favorite_child(adventurousness), self.choose_favorite_child(0.0), max(self.children, key = lambda child: child.visits)]
        
        elif last_choice == "reward": #Experimental   
            #uncomment when nesting is updated!
            #self.expand_full_path()
            return max(self.children, key = lambda child: child.maximum_reward)
        
    def update_maximum_reward(self, new_reward, new_path):
        if new_reward > self.maximum_reward:
            self.maximum_reward = new_reward
            self.maximum_path = new_path

    def expand_full_path(self):
        #expand tree along a path fully to save good rollouts        
        Current_node = self
        for move in self.maximum_path:
            if move not in Current_node.possible_moves:
                Current_node = Current_node.choose_particular_child(move)
            else:
                Current_node = Current_node.expand_given_move(move)
        Current_node.backpropagation(self.maximum_reward)
        
        

    def sp_selection_term(self, D=100):
        #Add extra term to UCB from Schad et al. Single-player Monte-Carlo tree search for SameGame (2011)
        #For this to work all the scores for each node are to be saved into a list, say self.scores_list
        #D is a high constant to be tuned. It makes the effect of this term less (more) relevant for children visited less often
        
        scores_sq = sum([score**2 for score in self.scores_list])
        return np.sqrt((scores_sq - self.visits*self.ucb1_score**2+D)/(self.visits))
        
        pass