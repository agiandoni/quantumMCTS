# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:01:09 2022

@author: Andoni Agirre
"""

import numpy as np

class Hand():
    def __init__(self):
        self.hand0 = np.ones((2))
        self.hand1 = np.ones((2))
        self.turn = 0
        self.winner = None
        self.h0_splittable = False
        self.h1_splittable = False
        #self.fussionable = True
        
    def draw_hands(self):     
        print("-----")
        print("PLAYER:",1)
        print(self.hand0)
        print("-----")
        print("PLAYER:",2)
        print(self.hand1)
        print("-----")
    
    def is_splittable(self):
        splittable = [0,0]
        if (self.hand0[0]==0 and self.hand0[1]%2==0) or (self.hand0[1]==0 and self.hand0[0]%2==0):
            splittable[0] == 1
        if (self.hand1[0]==0 and self.hand1[1]%2==0) or (self.hand1[1]==0 and self.hand1[0]%2==0):
            splittable[1] == 1
        return splittable
    
    def get_move_from_player(self):
        #Split/Fusion moves missing!
        self.draw_hands()
        if self.splittable[self.turn]:
            print("Split move is available!")
            print("Write 'X' to split!")
        player_hand = input("Choose one of your hands (1 or 2)")
        if player_hand == "X":
            self.turn = 2 if self.turn==1 else 1
            return ["X","X"]
        opponent_hand = input("Choose one of your oponent's hands (1 or 2)")
        if opponent_hand == "X":
            self.turn = 2 if self.turn==1 else 1
            return ["X","X"]
        move=[player_hand, opponent_hand]
        self.turn = 2 if self.turn==1 else 1
        return move
    
    def cycle(self):
        for hand in self.hand0:
            if hand > 5:
                hand = hand % 5
        for hand in self.hand1:
            if hand > 5:
                hand = hand % 5
        
    def play_move(self,move,who):
        #self.state(self.turn,move(1))+=self.state(self.turn,move(0))
        if move == ["X","X"]:
            if who==1:    
                self.hand0[:]=max(self.hand0)/2
                pass
            if who==2:
                self.hand1[:]=max(self.hand1)/2
                pass
        self.state[self.turn,move(1)]=(self.state[self.turn,move(1)]+self.state[self.turn,move(0)])%5
        self.cycle()
        
    def is_it_game_over(self):
        if self.hand0[0,:] == 0:
            self.winner = 1
        if self.shand1[1,:] == 0:
            self.winner = 0
            
            
hand = Hand()
hand.draw_hands()