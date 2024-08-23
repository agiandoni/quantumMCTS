# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:00:44 2022

@author: Andoni Agirre
"""

import numpy as np
import math
import copy
#from timeit import default_timer as timer
#from qzero_game import Qboard
#adventurousness = 1.41 #How much the algorithm will venture into unexplored paths!

def get_game_value( board_state, player ):
    winner = board_state.check_winner()
    assert(winner != None)
    return winner*player


class Node():
    def __init__(self, board_state, parent=None, minimum_reward = -np.inf, last_action_index=None):
        self.visits = 0 #counts how many times the node has been visited during a simulation
        self.score = 0.0 #keeps the total score of the node
        self.fully_expanded = False #Initializes node as a leaf node
        self.board_state = board_state #This will store the state of our game; ie where we are on the game tree. It comes from some other code
        self.children = [] #The nodes to which we can go from the current node will be stored here; unexpanded for now (list of Nodes)
        self.parent = parent #The parent of the current node (this takes a Node() class object)

        self.possible_moves = self.board_state.moves_in_position()        
        self.possible_moves = dict(zip(self.move_to_index(self.possible_moves), self.possible_moves))
        
        
        self.maximum_reward = minimum_reward #The score of the best rollout in the entire turn
        self.maximum_path = [] #Path memorized from root to the best scoring node found in the entire turn (in indices!!)
        
        self.action_index = last_action_index #How we got to this node, important for expanding path in modified MCTS!
        
        #For D term, start filling the list here at each cycle
        self.scores_list = []
        
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

    def choose_favorite_child(self, adventurousness, D_term = False):
        #adventurousness determines how much the algorithm will venture into unexplored paths!
        assert(len(self.children) != 0)
        
        if not D_term:
            best_child_index = np.argmax([(child.score / child.visits) + adventurousness * np.sqrt(np.log(self.visits) / child.visits) for child in self.children])
            best_child = self.children[best_child_index]     
            #assert(best_child == max(self.children, key=lambda child: child.ucb1()+adventurousness*child.ucb2()))
            return max(self.children, key=lambda child: child.ucb1()+adventurousness*child.ucb2())
        
        elif D_term:
            best_child_index = np.argmax([(child.score / child.visits) + adventurousness * np.sqrt(np.log(self.visits) / child.visits) + child.sp_selection_term() for child in self.children])
            best_child = self.children[best_child_index]
            return best_child
        
    def choose_particular_child(self, move):
        assert(len(self.children) > 0), "choose_particular_child() called on childless node"
        Chosen_child = None
        for child in self.children:
            if np.isclose(child.board_state.state[-1], move):
                Chosen_child = child
        
        assert(Chosen_child is not None), "Wanted child was not found by choose_particular_child()"
        return Chosen_child
    
    def choose_particular_child_from_index(self, move_index):
        assert(len(self.children) > 0), "choose_particular_child() called on childless node"
        Chosen_child = None
        for child in self.children:
            if move_index==child.action_index:
                Chosen_child = child
                return Chosen_child
        
        assert(Chosen_child is not None), "Wanted child was not found by choose_particular_child()"
        return Chosen_child
                


   # --- Expansion stage methods  ---

    def add_child(self, child):
        self.children.append(child)


    def expand(self):

        index, move = self.possible_moves.popitem() #Take only one of the unexpanded along with its index
                
        new_board = copy.deepcopy(self.board_state)
        new_board.make_move(move)

        child = Node(board_state=new_board, parent=self, last_action_index=index) #Initialize the new node
        self.add_child(child)

        # Also need to check whether all the children have been expanded
        if len(self.possible_moves) == 0:
            self.fully_expanded = True

        return child #Travels to the newly expanded child, ready for rolloout

    
    def expand_given_move(self, move_index):

        move = self.possible_moves.pop(move_index) #removes the key from dictionary
        
        new_board = copy.deepcopy(self.board_state)
        new_board.make_move(move)

        child = Node(board_state=new_board, parent=self, last_action_index=move_index) #Initialize the new node
        self.add_child(child)

        # Also need to check whether all the children have been expanded
        if len(self.possible_moves) == 0:
            self.fully_expanded = True

        return child #Travels to the newly expanded child, ready for rolloout
    
   # --- Rollout stage methods ---

    def random_step_from_position(self):
        #Deprecated!
        moves = self.board_state.moves_in_position()
        move = np.random.randint(len(moves))
        return moves[move]

    def rollout_policy(self, moves):
        move = np.random.randint(len(moves))
        return moves[move], move


    def rollout(self, player): #,position): #position is a Board object
        current_state = copy.deepcopy(self.board_state)
        indices_to_reward = []

        while not current_state.is_it_over():
            legal_moves = current_state.moves_in_position()

            legal_moves = dict(zip(self.move_to_index(legal_moves), legal_moves))
            move, index = self.rollout_policy(list(legal_moves.values()))
            indices_to_reward.append(index)
            current_state = current_state.make_move(move)

        return get_game_value(current_state, player), indices_to_reward


   # --- Backpropagation stage methods ---

    def backpropagation(self, end_score, alternating_rewards = False, who_played_last=None, moves_list=[], from_where=None):
        if alternating_rewards: 
            #For 2 player games. get_game_value() already fixed the first end_score relative to the MCTS player, then we just alternate signs
            self.visits += 1
            self.score += (end_score*who_played_last) #We do -turn because -player played to get to the node where it is player to play
            if self.parent:
                self.parent.backpropagation(end_score, True, -who_played_last)
        
        else:
            self.visits += 1
            self.score += end_score
           
            #SINGLE PLAYER REWARDS (only usable if rewards are NOT to alternate!)
            self.update_maximum_reward(end_score, moves_list, from_where)
            if self.parent:
                self.parent.backpropagation(end_score, moves_list=moves_list, from_where=from_where)
            
            
            

##################


    def travel(self, adventurousness = math.sqrt(2), D_term = False): #Modified so as to use the new expansion method
        #Tree policy of algorithm
        current_node = self

        while not current_node.board_state.is_it_over():
            if current_node.fully_expanded:
                current_node = current_node.choose_favorite_child(adventurousness, D_term=D_term)
            else:
                current_node = current_node.expand()

                #This loop should only terminate after a new node is expanded or a game over state is reach before expanding
                return current_node

        return current_node #This will execute if we reach a game over state during selection

    
    def MCTS_step(self, adventurousness: float, repeats: int, player: int, alternating_rewards= False, progress_intervals = 10, last_choice="max", D_term = False):
        #Chooses the move to play
        #start=timer()
        #Uncomment box for progress intervals
        ################################################################
        # if progress_intervals:
        #     print(f"Simulation/{repeats}: ",end="")

        for simulation in range(repeats):            
        #   print(simulation+1, timer()-start)
        #
        #   if progress_intervals:
        #       if simulation+1==repeats:
        #           print(f"{simulation+1}.")
        #       elif (simulation+1) % (repeats/progress_intervals) == 0:
        #           print(f"{simulation+1}, ", end="")
        ###############################################################
            
            # Expand until a leaf node is reached
            destination = self.travel(adventurousness, D_term)
            destination.board_state.fix_turn() #unnecessary for 1P games/puzzles
            # Rollout until a game over node is reached
            end_score, moves_list = destination.rollout(player)           
            destination.backpropagation(end_score, alternating_rewards, -destination.board_state.turn*player, moves_list=moves_list, from_where=destination)
           
        #Choose final move!
        if last_choice == "ucb":
            print("MCTS selected a move with a test-criterion! (Do not trust it)")
            return self.choose_favorite_child(adventurousness)
        
        elif last_choice == "max":
            return self.choose_favorite_child(0.0)
        
        elif last_choice == "robust":
            return max(self.children, key = lambda child: child.visits)
        
        elif last_choice == "compare":
            return [self.choose_favorite_child(adventurousness), self.choose_favorite_child(0.0), max(self.children, key = lambda child: child.visits)]
        
        elif last_choice == "reward":
            self.expand_full_path()
            return max(self.children, key = lambda child: child.maximum_reward)
        
    def update_maximum_reward(self, new_reward, new_path, from_where):
        if new_reward > self.maximum_reward:
            self.maximum_reward = new_reward
            self.maximum_path = copy.deepcopy(new_path)
            assert(len(self.maximum_path)+len(from_where.board_state.state)==self.board_state.total_turns), "PATH ERROR"
            self.from_where = from_where #where the expansion of max_reward should start from (Node)



    def expand_full_path(self):
        #expand tree along a path fully to save good rollouts        
        Current_node = self.from_where
        for index in self.maximum_path:
            if index not in Current_node.possible_moves: #if key does not exist in dict
                #THIS CAN ONLY BE TRIGGERED IF THE MAX REWARD ROLLOUT WAS PERFORMED BEFORE A SUBSEQUENT EXPANSION OF THE DESTINATION NODE!
                Current_node = Current_node.choose_particular_child_from_index(index)
                
                #Is this necessary?
                Current_node.score += self.maximum_reward
                Current_node.from_where = self.from_where
            else:
                Current_node = Current_node.expand_given_move(index)
                Current_node.score += self.maximum_reward
                Current_node.visits += 1
                Current_node.maximum_reward = self.maximum_reward #very important
                Current_node.from_where = self.from_where #If no better candidate is found in subsequent turns, we still need the complete information of the previous best run

        #Current_node.backpropagation(self.maximum_reward)
        
        

    def sp_selection_term(self, D=10000):
        assert(True==False)
        #Add extra term to UCB from Schad et al. Single-player Monte-Carlo tree search for SameGame (2011)
        #For this to work all the scores for each node are to be saved into a list, say self.scores_list
        #D is a high constant to be tuned. It makes the effect of this term less (more) relevant for children visited less often
        return np.sqrt(np.var(self.scores_list)+D/self.visits)
    
    


    def move_to_index(self, legal_moves):
        return list(range(len(legal_moves)))