# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:59:29 2022

@author: Andoni Agirre
"""
import numpy as np
import copy

class Qboard():
    def __init__(self, annealer, scoring_criterion = "energy", num_turns = 3, noise = "none", noise_strength = 0, SSR = False, SSR_previous_best=None, exponent = 2, delta = 0, b=30, maxcut_symmetries = False, gradient_at_final=False):
        """
        This is an implementation of the quantum annealing game (proposed by Chen et al).
        This is a one player game.
        The player needs to build an annealing path by choosing a number at each turn.
        The game then outputs a number according to the scoring criterion.
        The player must try to maximize said number by choosing the values that approach the optimal schedule as much as possible.
        
        Now it also includes a QAOA version, where the player has to build QAOA schedules in a similar way
        
        Parameters
        ----------
        annealer : Annealer
            The annealer which will try out the paths chosen by the player.
            Or the QAOA parameters.
        scoring_criterion : str, optional
            Specifies the way the game will score a given path. 
            The default is "energy".
            Choose from "energy", "fidelity", "test" (for testing purposes).
        num_turns : int, optional
            Fixes the number of turns in the game. The default is 3.
            (The allowed values must be specified in the self.num_choices and self.span attributes).
            For QAOA this is automatically fixed with the specified circuit depth

        """
        
        self.annealer = annealer
        if self.annealer.annealer_type == "QAOA":
            self.type = "QAOA"
            self.total_turns = 2*self.annealer.P
            
        else: #The QA case
            self.type = annealer.sPath.s_type #Specifies the gamemode
            if self.type == "tstep":
                self.num_choices = 20 #This specifies how we discretize the choice for s.
            elif self.type == "fourier":
                self.w_steps = 40 #How many values we can choose.
                self.span = 0.2       #How far from 0 (in absolute value) the value can go.
            elif self.type == "MCTS_test":
                self.num_choices = 2
                self.total_turns = 2
        
        self.criterion = scoring_criterion
        self.state = []
        self.game_over = False
        
        if self.type != "QAOA":
            self.total_turns = num_turns #Number of total moves of a game. 
        

        self.moves = self.build_moves_list(b, maxcut_symmetries)
        self.moves_played = 0
        
        self.noise_type = noise
        if noise == "classical":
            self.noise_std = noise_strength #Classical measurement noise strength
       

        self.exponent = exponent #For exponential reward mapping
        
        
        if not SSR:
            self.has_custom_rules = False
        else:
            self.has_custom_rules = True
            self.previous_best = SSR_previous_best
            self.generate_new_QAOA_searchspace(SSR_previous_best, delta=delta) # self.search_space <- dictionary with {key: turn number, value: [first, last, step]}
        
        self.gradient_at_final = gradient_at_final
        
        self.turn = 1 #This will never change as this is a 1P game, but it is a required attribute for the MCTS to work.

        
        
    def draw_board(self, fancy = False):
        print(self.state)
        if fancy:
            for index, move in enumerate(self.state):
                print("[", end="")
                if index <= self.moves_played:                    
                    for possibility in self.moves:
                        if move == possibility:
                            print("X", end="")
                        else:
                            print("-", end="")
                else:
                    print("HEMEN")
                    print(index, self.moves_played)
                    for possibility in self.moves:
                        print("-", end="")
                print("]")
                
    def parameters_to_dict(self):
        args_dict = {}
        
        for index, parameter in enumerate(self.state):
            args_dict[f"c{index}"] = parameter
        assert(len(args_dict) == self.total_turns == len(self.state))
        return args_dict
    
    def check_winner(self, exponentiate_energy=True, force_noiseless = False, max_energy=1.0, gradient_at_final = False):

        if self.type == "MCTS_test":
            if self.state == [0,0]:
                return np.exp(0)
            elif self.state == [0,1]:
                return np.exp(-2)
            elif self.state == [1,0]:
                return np.exp(-1)
            elif self.state == [1,1]:
                return np.exp(-0.5)
            
            
        if self.annealer.annealer_type == "QAOA":
            coefficients = self.generate_qaoa_friendly_state()
            if not self.gradient_at_final:
                results = self.annealer.run(coefficients)
            else:
                coefficients = [item for sublist in coefficients for item in sublist]
                
                results = [[self.annealer.gradient_optimize(coefficients)[0]]] #Energy after doing a MCTS+BFGS run; double [[]] so that it works with [0][-1] later
        
        
        elif self.annealer.annealer_type == "QA":
            results = self.annealer.anneal(self.parameters_to_dict())
        
        
        
        criterion = self.criterion
        if criterion == "energy":
            energy = results[0][-1]
            score = -np.abs(energy)
        elif criterion == "fidelity":
            fidelity = results[1][-1]
            score = fidelity
        elif criterion == "test":
            score = +sum(self.state)
        
        if self.noise_type == "classical" and not force_noiseless:
            noise = np.random.normal(0, self.noise_std)
            score += noise

            if score > 0: #because we are checking negatives of energy
                score = 0
        
        elif self.noise_type == "quantum" and not force_noiseless:
            if self.annealer.annealer_type=="QA":
                noise = self.annealer.generate_quantum_noise(self.parameters_to_dict())
            elif self.annealer.annealer_type=="QAOA":
                noise = self.annealer.generate_quantum_noise(self.generate_qaoa_friendly_state())
            score += noise
            if score > 0: #because we are checking negatives of energy
                score = 0
            
        if exponentiate_energy and criterion=="energy": #This should go after noise calculations!
            a = self.exponent
            score = np.exp(a*score) #Puts all negative values in between 0 and 1
            
        return score
    
    def is_it_over(self):
        return self.moves_played == self.total_turns
    
    def make_move(self, move):
        self.state.append(move)
        self.moves_played += 1
        return self
    
    def create_tstep_schedule(self, M):
        #Creates dsicretized linear schedule with M segments, taking the average value of s in each segment 
        args_keys=[f"c{i}" for i in range(M)]
        args_values = [i/M+1/(2*M) for i in range(M)]
        args = dict(zip(args_keys, args_values))
        return args
        
    def build_moves_list(self, b, maxcut_symmetries=False):
        #Returns the possible moves in a position barring some special rule.
        if self.type == "tstep":
            args=self.create_tstep_schedule(self.num_choices)
            return list(args.values())
        
        elif self.type == "tstep_old":
            step_size = 1/float(self.num_choices-1)
            moves = np.arange(0, 1, step_size)
            moves = np.append(moves, 1.0)
            return moves
            
        elif self.type == "fourier":
            step_size = 2*self.span / float(self.w_steps - 1)
            moves = np.arange(-self.span, self.span, step_size)
            moves = np.append(moves, self.span)
            return moves
        
        elif self.type == "QAOA":
            choices_num = b
            if maxcut_symmetries:
                moves = np.linspace(-0.5*np.pi, 0.5*np.pi, choices_num) #we don't want zero
            else:
                moves = np.linspace(-np.pi, np.pi, choices_num + 1)
                
            return moves[:-1] #Because 0 is the same as 2*pi
        
        
        elif self.type == "MCTS_test":
            return [0,1]
        
    def moves_in_position(self, fig2 = False, shuffle=False, custom_search_space = False, test_simple=False):
        
        if not self.has_custom_rules:
            legal = copy.deepcopy(self.moves)
            if self.moves_played == 0 and test_simple:
                return [0.425, 0.025]

            if self.type == "QAOA" and self.moves_played == 0 and not self.has_custom_rules: #Make use of inversion symmetry to limit one parameter
                #Do not shuffle before applying the symmetry! This rests on the fact that the moves are ordered!
                legal = list(legal)
                legal = legal[len(legal)//2:]

            legal = list(legal) #to list because mcts uses .pop() on these
            
            if fig2 == True: #To recreate a particular figure
                if self.type == "QAOA":
                    print("Warning: fig2 settings called for QAOA")
                legal = np.linspace(-0.2, 0.19, 40)
                legal= list(legal)
            return legal
        
        
        else:
            if self.moves_played == self.total_turns:
                return []
            first, last, n_choices = self.search_space[self.moves_played+1]
            return list(np.linspace(first, last, n_choices))

    
    def update_state(self, new_state):
        self.state = new_state
        self.moves_played = len(self.state)
        
    def generate_qaoa_friendly_state(self, params = None):
        # we want to go from [gamma1, beta1, ..., gammaP, betaP]
        # to [[gamma1, ..., gammaP], [beta1, ..., betaP]]
        if params is None:
            state = copy.deepcopy(self.state)
        else:
            state = params
        gammas = state[0::2]
        betas = state[1::2]
        
        #This assertion might be unecessary as self.total_turns is now hardcoded for QAOA when initializing game
        assert(len(gammas) == len(betas)), f"QAOA parameter mismatch (g{len(gammas)}, b{len(betas)}; unable to get reward"
        
        return [gammas, betas]

    def generate_classical_noise(self, noise_strength=None):
        return np.random.normal(0, noise_strength if noise_strength is not None else self.noise_strength)
    
    def generate_quantum_noise(self):
        #Generates quantum measurement noise, where strength is determined from the varaince of Hamiltonian (like in Yao et al.)
        return self.annealer.generate_quantum_noise(self.parameters_to_dict())
    
    def generate_new_QAOA_searchspace(self, previous_solution, delta = 0.05):
        #Should give a dictionary {turn_number: [first, last, n_choices], ...} #Odd turns gamma, even turns beta
        starting_beta  = 0.5*np.pi
        starting_gamma = 0
        last_beta = 0
        last_gamma = 0.5*np.pi
        n_choices = 30
        
        gammas, betas = self.generate_qaoa_friendly_state(previous_solution)
        gammas = gammas[0]
        betas = betas[0]
        gammas.insert(0, starting_gamma) 
        gammas.append(last_gamma) 
        betas.insert(0, starting_beta) 
        betas.append(last_beta) 


        
        search_space = {}
        index = 0
        for turn in range(self.total_turns):
            if (turn+1) % 2 == 0: #We choose a beta
                search_space[turn+1] = [betas[index]*(1-delta), betas[index+1]*(1+delta), n_choices]
                index+=1
            else: #We choose a gamma
                search_space[turn+1] = [gammas[index]*(1-delta), gammas[index+1]*(1+delta), n_choices]
        self.search_space = search_space
 
    
    def force_new_criterion(self, criterion):
        self.criterion = criterion
    
        
    def fix_turn(self):
        #It is safer if this exists
        pass
    
    
    
    
    
    