import math
import copy
import numpy as np

optimal = [-math.inf, []]
energies = np.array([])
filenames = [str(i) for i in range(11,0,-1)]
def brute_force(board, checkpoints = False, histogram = False, save_dir = "/home/projects/ku_00067/scratch/mcts_tests/"):
    global optimal
    global energies
    global filenames
    en_thr = 0.1
    if board.is_it_over():
        #If the game has ended, check if the path gives a better result
        outcome_to_check = [board.check_winner(), copy.deepcopy(board)]
        if -outcome_to_check[0] < en_thr:
            energies = np.insert(energies, 0, -outcome_to_check[0])
        if outcome_to_check[0] > optimal[0]:
            optimal = outcome_to_check
            return optimal
    else:
        for move in board.moves_in_position():
            board.make_move(move) #Make move
            brute_force(board, checkpoints = checkpoints, histogram = histogram, save_dir = save_dir) #Play next layer of moves
            
            if checkpoints and board.moves_played == 2:
                #print(f"{int(board.state[1]*10 + 1)}/{board.s_choices + 1}")
                # print(f"Explored up to {board.state}")
                # print(f"Best path so far: {optimal[1].state}, -<Hp> = {optimal[0]}")
                # print("")
                if histogram:
                    print(filenames)
                    name = "energies"+filenames.pop()
                    np.save(save_dir+name, energies)
                    energies = np.array([])
                #WRITE ENERGIES TO FILE AND SET IT TO EMPTY AGAIN
                
            del board.state[-1]      #Undo move
            board.moves_played -= 1  #Fix turn index
    return optimal
    
