# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:18:01 2023

@author: Andoni Agirre

Simple template for building a game for MCTS
"""

class TestBoard():
    def __init__(self, n_turns):
        self.n_turns = n_turns
        self.total_turns = n_turns
        self.state = []
        self.turn = 1
   
    def is_it_over(self):
        return (len(self.state) >= self.n_turns)
        
    def fix_turn(self):
        pass
    
    def check_winner(self):
        if self.is_it_over():
            return sum(self.state)
        else:
            return None
    
    def moves_in_position(self):
        if not self.is_it_over():
            return [-1.1,2.1]
        else:
            return []
   
    def make_move(self, move):
        self.state.append(move)
        return self