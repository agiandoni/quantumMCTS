# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:36:09 2022

@author: Andoni Agirre
"""

""""
We have n bits, m 3-bit clauses.

The 3SAT problem will be read from a file:

for example, a file could read

########### 3sat.dat #################
1 -2 3 1 2 -4 -4 5 -6
######################################

the 3SAT statement it represents is
(x1 OR NOT x2 OR X3) AND (X1 OR X2 OR NOT X4) AND (NOT X4 OR X5 OR NOT X6);
the number is the index of the bit it references (maybe start from 0),
the negative sign means its negation enters the clause.

We can get m from 1/3*len() of the file data (m=3 in this case); 
n from the |max()| of the data (n=6 in this case).
We want to save the problem as an array separated into clauses
[[1, -2, 3], [1, 2, -4], [-4, 5, -6]]


Alternatively, the file could instead be of the form

########### 3sat.dat #################
1 -2 3
1 2 -4                                            <---- I ENDED UP DOING THIS ONE!
-4 5 -6
######################################

to make it easier to read?



NOTES: The algorith works. BUT ONLY if the bits in each clause (so each line in the data file) is written in ascending order in absolute value;
this is fine because the or operation commutes. But the read_problem_statement() method of the class should still sort each row.
"""

import numpy as np
import itertools
import qutip as qt


def sigma(i, N):
    """
    Creates the 2^Nx2^N dimensional tensor product with N-1 2x2 identity operators and one sigmaz in the ith position,
    sigmaz_i = 1 (x) ... (x) sigmaz (x) 1 (x) ... (x) 1

    Parameters
    ----------
    N : INT
        2^Number_of_bits
    i : INT
        The position of sigmaz()

    """
    operators = []
    for index in range(1, N+1):
        if index == i:
            operators.append(qt.sigmaz())
        else:
            operators.append(qt.identity(2))
    return qt.tensor(operators)
    
    
    
class SAT3:
    def __init__(self, filename):
        print("Loading 3-SAT problem...")
        self.statement = self.read_problem_statement(filename)
        self.bit_signs = -np.sign(self.statement) #Reverse the signs to add them normally
        print("The statement is:\n", self.statement)
        print("Reverse sings:\n", self.bit_signs)
        self.M = self.get_M()
        print("No of clauses, M: ", self.M)
        self.N = self.get_N()
        print("No of bits", self.N)
        self.alpha = self.M / self.N
        
        self.prefactor = 1/(2**3) #1/8 for 3-SAT
        self.h_term = {} #Linear term in H, it goes with the individual spins
        self.J_term = {} #2-spin interaction term, there will be at most N different terms
        self.X_term = {} #3-spin interaction term, there will be at most M different terms
        self.Hamiltonian = 0
            
    def get_M(self):
        #Get the number of clauses of the 3SAT problem
        assert(self.statement.size%3==0), "Each clause needs to have three literals"
        return len(self.statement)
        
        
    def get_N(self):
        #Get the number of bits of the 3SAT problem
        return np.abs(self.statement).max()
    
    def read_problem_statement(self, filename): #I SHOULD SORT THE BITS FOR EACH CLAUSE HERE! THE OR OPERATION COMMUTES ANS IT HELPS A LOT IN THE CALCULATIONS
        #read 3SAT from file
        return np.loadtxt(filename, dtype=int)
        # problem = np.loadtxt(filename, dtype=int)
        # order_indices = np.argsort(np.abs(problem))
        # problem_in_order = np.take_along_axis(problem, order_indices, axis = 1)
        # return problem_in_order
        
    def calculate_H_parameters(self):
        pair_counts = {} #How often each pair shows up in the 3SAT problem (signed), {12: -1, 13: 2, ...}
        triplet_counts = {}
               
        for i, clause in enumerate(zip(abs(self.statement),self.bit_signs)): #The literal information (+ or -) is lost from self.statement; but it was stored in bit_signs!
            
            #free terms
            for bit in zip(clause[0],clause[1]):
                if bit[0] not in self.h_term:
                    self.h_term[bit[0]] = 0
                self.h_term[bit[0]] += bit[1]


            #Spin pair terms
            pairs = (list(itertools.combinations(clause[0], 2)))
            pairs_signs = list(itertools.combinations(clause[1], 2))
            pairs_signs = list(p[0] * p[1] for p in pairs_signs) #This is what we sum to the corresponding term
                        
            for j, pair in enumerate(zip(pairs,pairs_signs)):
                if pair[0] not in pair_counts:
                    pair_counts[pair[0]] = 0
                    
                pair_counts[pair[0]] += pair[1]
                      
                
            #Spin triplet term
            triplet = tuple(clause[0]) #There is only one triplet per clause, so this is simpler than the pairs
            triplet_sign = np.prod(clause[1]) #This is what we sum to the corresponding term
  
            if triplet not in triplet_counts:
                triplet_counts[triplet] = 0

            triplet_counts[triplet] += triplet_sign
  
        self.J_term = pair_counts
        self.X_term = triplet_counts
        
        
    
    def create_problem_Hamiltonian(self, parameters_precalculated = False):
        if parameters_precalculated:
            self.calculate_H_parameters()
        
        H = 0
        
        for index in self.h_term:
            H += self.h_term[index]*sigma(index, self.N)
        
            
        for indices in self.J_term:
            H += self.J_term[indices] * sigma(indices[0], self.N) * sigma(indices[1], self.N)

        
        for indices in self.X_term:
            H += self.X_term[indices] * sigma(indices[0], self.N) * sigma(indices[1], self.N) * sigma(indices[2], self.N)
            
            
        #H = qt.Qobj(H.data.reshape(2**self.N,2**self.N))
        identity_composite = sigma(0,self.N)
        return self.prefactor*(H+ self.M* identity_composite)
        #For key,value in self.h_term:
        #   if value is not 0:
        #       Generate the corresponding operator indicated by the key via tensor products (*)
        #       Add it into H with the corresponding coefficient given by value
        #Create similar loops for the other terms
        #
        #
        #
        #(*) Not so simple! 
    
    def createInitialH(self):
        #We create the transverse field Hamiltonian
        nbits=2
        H0 = 0
        i2x2 = qt.identity(2)
        literal = 0.5*(i2x2 - qt.sigmax())
        
        for bit1 in range(nbits):
            operator_list = [literal if bit2 == bit1 else i2x2 for bit2 in range(self.N)]
            H0 += qt.tensor(operator_list)
    
        H0 = qt.Qobj(H0.data.reshape(4,4))
        psi0 = H0.groundstate()
        return H0, psi0
        
problem1 = SAT3("problem.txt")        
problem1.calculate_H_parameters()
H=problem1.create_problem_Hamiltonian()
        
        
        
        
        
        
        