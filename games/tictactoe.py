# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:28:43 2022

@author: Andoni Agirre

Simple tic tac toe implementation for MCTS to play
"""
import numpy as np

class Board():
    def __init__(self):
        self.state = np.zeros((3,3))
        self.turn = 1
        self.game_over = self.is_it_over()

    def draw_board(self):
        #matching_symbol = lambda number: ("O" if number == 1 else ("X" if number == -1 else " "))
        def matching_symbol(number): return (
            "O" if number == 1 else ("X" if number == -1 else "-"))
        print("   1 2 3")
        for i,row in enumerate(self.state):
            print(i+1," ", end="")
            for element in row:
                print(matching_symbol(element),end=" ") #do not break line until the whole row is on console
            print("\n")

    def check_winner(self):
            #check rows
            for row in self.state:
                if np.sum(row) == 3:
                    return 1
                if np.sum(row) == -3:
                    return -1
            #check columns
            for column in np.transpose(self.state):
                if np.sum(column) == 3:
                    return 1
                if np.sum(column) == -3:
                    return -1
            #check diagonals
            if np.trace(self.state) == 3:
                return 1
            if np.trace(self.state) == -3:
                return -1
            if np.trace(np.flip(self.state, axis= 1)) == 3: #Looks cool but will this even work
                return 1
            if np.trace(np.flip(self.state, axis= 1)) == -3:
                return -1
            #check tie
            if 0 not in self.state:
                return 0 #TIE
            return None #Game keeps going

    def is_it_over(self):
        self.game_over= self.check_winner() != None
        return self.game_over

    def reward(self):
        return self.check_winner()


    def input_move(self): #"O" (1), the player, makes a move
        print("Make a move!")
        row = int(input("Input row (1 to 3): "))
        column = int(input("Input column (1 to 3): "))
        if self.state[row-1,column-1]== 0:
            return np.array([row-1,column-1])
            #return row-1,column-1
        print("-----")
        print("Input a legal move")
        row, column = self.input_move()
        return np.array([row,column])

    def make_move(self, move):
        # At this point, move should be legal already
        self.state[move[0],move[1]] = self.turn
        self.turn *= -1
        return self

    def moves_in_position(self):
        legal = []
        for row in range(3):
            for column in range(3):
                if self.state[row,column]==0:
                    legal.append([row,column])
        return legal

    def fix_turn(self):
        i = np.sum(self.state)
        self.turn = 1 if i==0 else -1


