# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:03:42 2022

@author: Andoni Agirre
"""
import numpy as np
import qutip as qt
import annealing

H0, psi = annealing.create_transverse_H_and_psi(2)
H1 = qt.tensor(qt.sigmaz(), qt.sigmaz())

class salsa():
    def __init__(self):
        self.f = self.ss
    def ss(self, t, args):
        result = t/5
        # for i in range(args["b"]):
        #     result += args["a"]
        return result
    
def s(t, args):
    return t/5

salsita = salsa()
#H = [[H0, lambda t, args: 1 - s(t, args)], [H1, s]]
#H = [[H0, lambda t, args: 1 - salsita.f(t, args)], [H1, salsita.f]]
H = [[H0], [H1-H0, salsita.f]]

#psi = qt.Qobj(1/2 * np.ones(4))
t=[1,2,3,4,5]
print(psi[1])
print(H)
results = qt.mesolve(H, psi, t, args = {"a": 1, "b": 2})



#https://groups.google.com/g/qutip/c/NAGU4iKZNBY
#I might want to, either:
#1.- Work with dim [2**N, 2**N] operators and dim [2**N, 1] states, which removes all information on how the tensor product was formed. 
#Maybe it is technically wrong (although I cannot concretely say why with my QM knowledge), 
#and the computer forgets about the qubits existed, but this is practically not a problem, just work with the matrix and things work.
#2.- Make a function that creates a basis; ie, for n-qubits: basis = [[2, ..., 2], [2, ..., 2]] operators and basis = [[2, ..., 2], [1, ..., 1]] states.
#Every A = qt.Qobj() I create will be fixed by A.dims = basis, or Qobj(np.array(a), dims = basis).
#This might be the proper way?