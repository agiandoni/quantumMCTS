# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:41:02 2022

@author: Andoni Agirre
"""
import numpy as np
import qutip as qt

def sigma(i, N, axis = "z"):
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
    if axis == "z":
        sigma_op = qt.sigmaz()
    elif axis == "x":
        sigma_op = qt.sigmax()
    elif axis == "y":
        sigma_op = qt.sigmay()
    else:
        raise Exception("Enter a valid axis value (x, y ,z)")

    operators = []
    for index in range(1, N+1):
        if index == i:
            operators.append(sigma_op)
        else:
            operators.append(qt.identity(2))
    return qt.Qobj(qt.tensor(operators))

    
def create_transverse_H_and_psi(N):
    #Creates the transverse field Hamiltonian and its ground state.
    H0 = 0
    idNN = qt.tensor([qt.identity(2) for i in range(1, N+1)])
    for bit in range(1,N+1):
        sigmanow = sigma(bit, N, axis = "x")
        H0 += idNN - sigmanow
    
    H0 = 0.5*H0
    psi0 = H0.groundstate()[1]
    return H0, psi0




class DimensionFixer():
    """
    Creates the required dimensionalities for Qutip object operations.
    Pass this as A = Qobj(a, dim = generate_bases(n_qubits)[0/1]),
    or, alternatively, fix it after object creation with A.dims = generate_bases(n_qubits)[0/1]
    Parameters
    ----------
    n_qubits : INT
        The number of qubits to use
        
    Returns
    operator_dims = [[2, ..., 2], [2, ..., 2]] for operators, and operator_dims = [[2, ..., 2], [1, ..., 1]] for states
    with len(*_dims[0/1]) = n_qubits
    """
    
    def __init__(self, n_qubits):
        self.N = n_qubits
        self.oper = self.a()
        self.state = self.oper[1]
        self.oper = self.oper[0]
        
    def a(self):
        operator_dims = [self.N*[2], self.N*[2]]
        state_dims = [self.N*[2], self.N*[1]]
        return operator_dims, state_dims

def qIdentity(n_qubits):
    return qt.tensor([qt.identity(2) for i in range(1, n_qubits+1)])


def read_3SAT_from_file(file_path, instance_num):
    #Should load a 3SAT problem from a file
    data = np.loadtxt(file_path)
    H_finals = []
    assert(instance_num <= data.shape[0]), f"There are not that many ({instance_num}) problem instances in the file, (max {data.shape[0]})"
    for i in range(instance_num):
        result = data[i,:].astype(int)
        counts = np.bincount(result)
        H_final = qt.Qobj(np.diag(counts))
        D = DimensionFixer(int(np.log2(len(counts))))
        H_final.dims = D.oper
        H_finals.append(H_final)
    return H_finals
    
def generate_simple_problem():
    H0, psi0 = create_transverse_H_and_psi(2)
    Hp = np.diag([0,1,2,1])
    Hp = qt.Qobj(Hp, dims = [[2,2],[2,2]])
    psip = Hp.groundstate()[1]
    return H0, Hp, psi0, psip

def generate_simple_problem2():
    H0, psi0 = create_transverse_H_and_psi(2)
    Hp = np.diag([0,1,1,1])
    Hp = qt.Qobj(Hp, dims = [[2,2],[2,2]])
    psip = Hp.groundstate()[1]
    return H0, Hp, psi0, psip

def generate_even_simpler_problem():
    H0, psi0 = create_transverse_H_and_psi(1)
    Hp = np.diag([0,1])
    Hp = qt.Qobj(Hp, dims = [[2],[2]])
    psip = Hp.groundstate()[1]
    return H0, Hp, psi0, psip

def linstep_dict(steps):
    path = {}
    for step in range(steps):
        path[f"c{step}"] = float(step)/(float(steps)-1)
    return path