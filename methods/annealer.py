# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:26:06 2022

@author: Andoni Agirre

Annealer and QAOA objects
"""
import numpy as np
import qutip as qt
from methods.quantum_methods import *
# from 3SatH import SAT3
from scipy.optimize import minimize
import math

class sPath():
    def __init__(self, s_type : str, T : float, times_num : int):
        """
        Object that stores the path along which to evolve a hybrid Hamitonian  H(t) = s(t)*H1 + (1-s(t))*H0, where H1 is the problem Hamiltonian whose ground state needs to be obtained.
        It also specifies the times at which the evolution of the state is outputted, used for visualization of the evolution along the path. 
        self.s(t, args) will be the path s(t) that characterizes the evolution.        
        self.draw_path(args) can be used to visualize the path at the specified times.
        
        Parameters
        ----------
        s_type : str
            Specifies how the annealing schedule will be built, args should be structured accoridngly
            "fourier": 
            "linear" :
            "tstep"  :
            "digital":
            
        T : float
            Sets the total annealing time.
            
        times_num : int
            The number of times within [0,T] that the evolved states will be outputted.

        """
        
        self.T = T
        self.steps = times_num
        self.times = np.linspace(0, self.T, self.steps)
        
        self.s_type = s_type
        self.generate_path()
        
        self.s_values = []
        
    def draw_path(self, args = {}):
        #Visualize the path for a configuration args, empty by default. Saved as list in self.s_values.
        for t in self.times:
            self.s_values.append(self.s(t, args))
            
    def generate_path(self, new_type = None):
        #Can be used to update s_type.
        s_type = self.s_type if new_type is None else new_type
        if s_type == "linear":
            self.s = self.linear
        elif s_type == "fourier":
            self.s = self.fourier
        elif s_type == "digital":
            self.s = self.digital
        elif s_type == "tstep":
            self.s = self.tstep
        elif s_type == "digital":
            self.beta = []
            self.gamma = []
            self.s = self.digital
        elif s_type == "tstep2":
            self.s = self.tstep2

    ##### Specific annealing functions #####
    def linear(self, t, args = {}):
        return t/self.T
    
    def linear_oneminus(self, t, args={}):
            return 1.0-t/self.T

    def fourier(self, t, args = {}):
        if t > self.T:
            return 1
        elif t < 0:
            return 0
        else:
            s = t/self.T
            for k, coefficient in enumerate(args):
                omega = np.pi*(k+1)/self.T
                s += args[coefficient] * np.sin(omega*t)
            return s
    
    def tstep(self, t, args):
        #the keys in args should go from "c0" to "cnum_turns-1", the values should be the discrete choices of s for each turn, in order;
        #e.g. {"c0":0.0, "c1":0.5, "c2":1.0} for a 3 turn game.
        
        num_turns = len(args) #The plateaus are in np.linspace(0, self.T, num_turns+1) (because we don't count the last step! IE linspace(0,10,3) is a 2 turn game!!). The step_size determines this.
        step_size = self.T / (num_turns)
        index = int(t/step_size)
        if index > num_turns-1: #-1 because the indices go from 0 to num_turns - 1.
            return 1 #The ode solver might require times > self.T, since the evolution is over, we should have s = 1 at those times.
        return args[f"c{index}"]
    
     
    def digital(self, t, args):
        pass
    ########################################
    

class Annealer():
    def __init__(self, H0, H1, psi, sPath, reference_state = None):
        self.H0 = H0
        self.H1 = H1
        self.psi = psi
        self.sPath = sPath
        
        if not reference_state:
            self.reference_state = H1.groundstate()[1] # .groundstate() returns [E0, psi0]
        else:
            self.reference_state = reference_state
      
        self.annealer_type = "QA" #For the game object to distinguish both types

    

    def anneal(self, coefficients = {}, times_list = None, extra_path_calculations = False):
        #coefficients is a dictionary {"c1":c1, ..., "cM":cM} of all of the desired coefficients to define s(t).
        #The type of self.sPath should be one such that it has a method that interprets the coefficients correctly.
        
        
        if self.sPath.s_type == "tstep":
            #We anneal manually if we have a Hamiltonian with jumps (this overwrites times_list to be the number of turns)
            return self.anneal_manual(coefficients, extras = extra_path_calculations)
            
        #The times at which the results will be outputted
        times_list = self.sPath.times if times_list is None else times_list
        
        #We build the hybrid Hamiltonian with s(t) governing its time evolution
        self.H = self.construct_hybrid_H()
        
        #We evolve the state
        results = qt.mesolve(self.H, self.psi, times_list, args = coefficients, options = qt.Options(nsteps=1000000))# max_step = 0.5))
        energy, fidelities = self.results_from_state_list(results.states, self.H1, self.reference_state)
        if not extra_path_calculations:  
            return energy, fidelities, results.states
        
        else:
            states = results.states
            M = 5 #The number of stored eigenvalues/eigenstates of the hybrid H
            energies_initial, fidelity_initial = self.results_from_state_list(states, self.H0, self.reference_state)
            energies_hybrid = []
            states_hybrid = []
            inst_energies = []
            inst_fidelities = []
            s = []
            for time in times_list:
                s.append(self.sPath.s(time, coefficients))
            for i, time in enumerate(times_list):
                H_t = qt.qobj_list_evaluate(self.H, time, coefficients)
                inst_energies.append(qt.expect(H_t, states[i]))
                evals, ekets = H_t.eigenstates(eigvals=M) #Evals has shape (M, len(times_list))
                energies_hybrid.append(evals) #list of arrays [array(E0(t0), E1(t0), ..., EM(t0)), ... array(E0(T), E1(T), ..., EM(T))]
                states_hybrid.append(ekets)
                inst_fidelities.append(np.abs(states[i].overlap(ekets[0]))**2)
            return energy, fidelities, s, energies_initial, fidelity_initial, energies_hybrid, states_hybrid, states, inst_energies, inst_fidelities

        
    
    
    def results_from_state_list(self, state_list, Hamiltonian, reference_state):
        energies = qt.expect(Hamiltonian, state_list)
        if self.reference_state:
            fidelities = []
            for state in state_list:
                fidelity = np.abs(state.overlap(self.reference_state))**2
                fidelities.append(fidelity)
            return energies, fidelities
        return energies, []

    def anneal_manual(self, coefficients, extras = False):
        N_turns = len(coefficients)
        delta_t = self.sPath.T / (N_turns) #The length of the plateaus
        
        self.H = self.construct_hybrid_H()
        
        if extras:
            H_list = []
        
        states = [self.psi]
        for i in range(N_turns):
            H_t = qt.qobj_list_evaluate(self.H, delta_t*i, coefficients)
            U = -1j*delta_t*H_t
            U = U.expm()
            new_state = U * states[-1]
            states.append(new_state)

            if extras:
                H_list.append(H_t)
        
        energy, fidelities = self.results_from_state_list(states, self.H1, self.reference_state)
        if extras:
            return energy, fidelities, states, H_list
        return energy, fidelities, states
            
        
    def anneal_digital(self):
        #Change sPath such that if type = "digital" is chosen, the sPath object itself works as a trotter splitted gamma, beta.
        state_list = [self.psi]
        for m in range(self.P):
            Uz = -1j*self.sPath.gamma[m]*self.H1
            Uz = Uz.expm()
            Ux = -1j*self.sPath.beta[m]*self.H0
            Ux = Ux.expm()
            state_list.append(Ux*Uz*state_list[m])
            
            energy, fidelities = self.results_from_state_list(state_list)
            return energy, fidelities
            
            
    
    def construct_hybrid_H(self, analog = True):
        #We build the path dependent Hamiltonian that will be used for evolution during the annealing
        if analog:
            H = [[self.H0, lambda t, args: 1 -self.sPath.s(t, args)],[self.H1, lambda t, args: self.sPath.s(t, args)]]
            return H
    
    def generate_quantum_noise(self, coefficients):
        scale = 1.0
        H = self.construct_hybrid_H()
        final_state = qt.mesolve(H, self.psi, [0, self.sPath.T], args = coefficients, options = qt.Options(nsteps=1000000)).states[-1]
        avg_en = qt.expect(self.H1, final_state)
        avg_en_sq = qt.expect(self.H1**2, final_state)
        quantum_std = math.sqrt(avg_en_sq-avg_en**2)
        return scale*np.random.normal(0, quantum_std)
    
############
class QAOA():
    def __init__(self, H0, H1, psi0, reference_state = None, qaoa_step_number = 1):
        self.H0 = H0
        self.H1 = H1
        self.psi0 = psi0
        self.P = qaoa_step_number
        if not reference_state:
            self.reference_state = H1.groundstate()[1] # .groundstate() returns a list [E0, psi0]
        else:
            self.reference_state = reference_state
        
        self.annealer_type = "QAOA"
        
        assert(psi0.shape[0] == reference_state.shape[0] == H0.shape[0] == H1.shape[0])
        self.n_bits = round(np.log(psi0.shape[0])/np.log(2))

        
   
    
    
    def run(self, coefficients, times_list = None, get_gradient = False):
        """
        Calculates the energy obtained with QAOA for a set of angles stored in coefficients 

        Parameters
        ----------
        coefficients : list of QAOA parameters [[gamma_1, ..., gamma_P],[beta_1, ..., beta_P]]
        times_list : int, optional (leftover from QA)
        get_gradient : Bool (False), optional
            If true the method also calculates the gradient of the QAOA ansatz from the analytic expression
        extra_path_calculations : Bool (False), optional
            leftover from QA, but will implement it in future (with QA equiv paths or similar)

        Returns
        -------
        energies_at_every_step, fidelities_at_every_step, states_at_every_step, gradient or empty_list

        """
        #We can apply the H0 part of the QAOA ansatz without having to exponentiate
        fast_x_rotate = True
        def fast_x_rotation(beta):
            sigmax_exp = qt.qeye(2)*np.cos(beta*0.5) + 1j*qt.sigmax()*np.sin(beta*0.5)
            operators = [sigmax_exp for n in range(self.n_bits)]
            x_evolution = qt.tensor(operators)
            return x_evolution
        
        states  = [self.psi0]
        for step in range(self.P):
            zphase = -1j*coefficients[0][step]*self.H1
            if not fast_x_rotate:
                xphase = -1j*coefficients[1][step]*self.H0
                states.append(xphase.expm()*zphase.expm()*states[step])
            else:
                states.append(fast_x_rotation(coefficients[1][step])*zphase.expm()*states[step])            
            
           
        energies, fidelities = self.results_from_state_list(states, self.H1, self.reference_state)

        
        gradient = 0
        if get_gradient:
            gamma_grad = []
            beta_grad = []
            #Sx = Qobj(jspin(self.num_qubits,op='x',basis='uncoupled').data.reshape([self.N,self.N]))

            state_Cm=[self.H1*states[-1]]
            for m in range(self.P):
                zphase = -1j*coefficients[0][-1-m]*self.H1
                state_Cm.append(zphase.expm(method='sparse')*fast_x_rotation(-coefficients[1][-1-m])*state_Cm[m])

            for m in range(self.P):
                gamma_grad.append(state_Cm[self.P-m].overlap(self.H1*states[m]))
                beta_grad.append(state_Cm[self.P-m-1].overlap(self.H0*states[m+1]))
            gradient = 2*np.concatenate((np.array(gamma_grad).imag, np.array(beta_grad).imag))

            return energies, fidelities, states, gradient
        
        return energies, fidelities, states, []
        
        
        
    def results_from_state_list(self, state_list, Hamiltonian, reference_state):
        energies = qt.expect(Hamiltonian, state_list)
        if self.reference_state:
            fidelities = []
            for state in state_list:
                fidelity = np.abs(state.overlap(self.reference_state))**2
                fidelities.append(fidelity)
            return energies, fidelities
        return energies, []
    
    
    def estimate_optimal_initial_parameters(self, method = "linearQA", max_step = 4*np.pi, num_times = 100):
        #This could be utilized to determine the required MCTS discr. to resolve the minimum
        if method == "linearQA":
            #Step discretized schedule and trotterized parameters
            def beta_k(k, delta_t):
                return (1 - k)*delta_t 
            def gamma_k(k, delta_t):
                return k*delta_t
            
        elif method == "linearQAp1":
            #Step discretized schedule and trotterized parameters +1??
            def beta_k(k, delta_t):
                return (1 - k)/(self.P+1)*delta_t 
            def gamma_k(k, delta_t):
                return k/(self.p+1)*delta_t
            
            
        energies = []
        time_steps = np.linspace(0, max_step, num_times)
        
        for time_step in time_steps:
            betas = [beta_k(k, time_step) for k in range(1, self.P+1)]
            gammas = [gamma_k(k, time_step) for k in range(1, self.P+1)]
            coefficients = [betas, gammas]
            energy = self.run(coefficients)[0][-1]
            energies.append(energy)
        
        best_time = time_steps[energies.index(min(energies))]
        optimal_betas = [beta_k(k, best_time) for k in range(1, self.P+1)]
        optimal_gammas = [gamma_k(k, best_time) for k in range(1, self.P+1)]
        return [optimal_betas, optimal_gammas], [best_time, min(energies)], energies
    
    def get_equivalent_QA_parameters(self, parameters):
        parameters = list(np.array(parameters).flatten())
        normalized_parameters = [i if i<=np.pi else i-2*np.pi for i in parameters] #In case we used [0,2*pi) ranged parameters
        equiv_annealing_time = np.sum(normalized_parameters)
        self.annealing_time = equiv_annealing_time
        return equiv_annealing_time

     
def gradient_optimize(annealer, init_coeffs = None, jacobian = True):
    
    P=annealer.P
    
    def get_reward(x):
        #x is flattened array [gamma1, ..., gammaP, beta1, ..., betaP]
        gammas = x[:P]
        betas  = x[P:]
        energies = annealer.run(coefficients = [gammas, betas])[0]
        return energies[-1]
    
    def calculate_ansatz_gradient(x):
        gammas = x[:P]
        betas  = x[P:]
        gradient = annealer.run(coefficients = [gammas, betas], calculate_gradient = True)[-1]
        return np.array(gradient)
    
    init_coeffs = 2*np.pi*np.random.rand(2*P) if init_coeffs is None else init_coeffs
    res = minimize(get_reward, init_coeffs, jac = calculate_ansatz_gradient if jacobian else None, method = "BFGS")
    n_runs, optimized_parameters, energy = res.nfev, res.x, res.fun
    
    return energy, optimized_parameters, n_runs
