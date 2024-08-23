import math
def minimax(new_board, player):
    if new_board.is_it_over():
        value= new_board.check_winner()
        return [None, value]
    
    if new_board.turn == player: #Turn of minimax; we want to maximize rewards!
        best = -math.inf
        for move in new_board.moves_in_position():
            new_board.make_move(move)
            value = minimax(new_board, new_board.turn)[1]
            if value > best:
                best = max(value, best)
                best_move = move
            #Undo the move
            del new_board.state[-1]
            new_board.moves_played -= 1

        return [best_move, best]
        
    else: #Turn of the oponent of minimax; we want to minimize rewards!
        best = -math.inf 
        for move in new_board.moves_in_position():
            new_board.make_move(move)
            value = minimax(new_board)[1]
            if value > best:
                best = max(value, best)
                best_move = move
            new_board.state[move[0],move[1]] = 0 #undo the move
            new_board.fix_turn()
            
        return [best_move, best]

def find_best_move(board, player = 1):
    res = minimax(board, player)
    return res