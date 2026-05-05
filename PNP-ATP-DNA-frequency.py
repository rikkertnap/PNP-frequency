import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig



# ---------------------------------------------------------
# Physical parameters 
# ---------------------------------------------------------

kB = 1.380649e-23
T = 298
e = 1.602e-19
NA = 6.022e23
eps0 = 8.854e-12

eps_r = 78
epsilon = eps_r * eps0

params = {
    # diffusion constants  coefficient (m^2/s)
    "D_M":1e-10,
    "D_A":3e-10,       #3e-10,
    "D_MA":3e-10,
    "D_D": 1e-11,
    "D_MB": 1e-11,

    # k_on and k_off rates 
    "k_on_D":1e7,           #  in 1/(Ms) 
    "k_off_D":1e4,          #  in 1/s
    "k_on_A": 1.44e8,       #  in 1/(Ms) 
    "k_off_A": 7e3,         #  in 1/s
    
    # concentrations in M
    "c_DNA":50e-3, # total concentration DNA  c_DNA = C_D +c_MB
    "c_M": 1e-3,
    "c_ATP": 4e-3, # total c_ATP = c_A + c_MA 
    "c_Na":0.15,
    "c_Cl":0.15,

    # charges in C
    "qM":2*e,
    "qA":-4*e,
    "qMA":-2*e,
    "qD": -2.0*e,
    "qMB": 0.0
}

print(params)


# ---------------------------------------------------------
# Reaction + transport
# ---------------------------------------------------------

def solve_DNA_reaction(params):
    """ 
        Returns : S_free and theta
        c_D = S_free: concentration of free phosphates that have not Mg bound
        c_MB    :  concentration of phosphates that has bound with Mg 
        theta_D : fraction of phospahte that is not bound with Mg
    """
    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]

    c_DNA = params["c_DNA"]   # S_) = c_DNA 
    c_M = params["c_M"]

    K_D = k_off_D / k_on_D

    theta_D = 1.0 / (1.0 + c_M / K_D)
    c_D = theta_D * c_DNA # S_0
    c_MB = (1-theta_D) * c_DNA 
    
    return c_D, c_MB, theta_D


def solve_ATP_reaction(params):
    """ 
        c_A     : concentration of free ATP not bound with Mg 
        c_MA    : concentration of ATP that has bound with Mg 
        theta_A : fraction of ATP  that is not bound with Mg
    """
    k_on_A = params["k_on_A"]
    k_off_A = params["k_off_A"]
    c_ATP = params["c_ATP"]
    c_M = params["c_M"]

    K_A = k_off_A / k_on_A
    theta_A = 1.0 / (1.0 + c_M / K_A)
    c_A = theta_A * c_ATP
    c_MA = (1.0 -theta_A) * c_ATP

    return c_A, c_MA, theta_A

def compute_coefficients(params):
    """ 
        Returns : a_D, B_D, a_A, B_A, alpha_A, alpa_D 
    """

    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    k_on_A = params["k_on_A"]
    k_off_A = params["k_off_A"]

    c_M = params["c_M"]

    c_D, _ , _ = solve_DNA_reaction(params)
    c_A, _ , _ = solve_ATP_reaction(params)

    a_D = k_on_D * c_D
    B_D = k_off_D          # k_on_D * c_M + k_off_D
    a_A = k_on_A * c_A
    B_A = k_off_A
    alpha_A = k_on_A * c_M
    alpha_D = k_on_D * c_M 

    return a_D, B_D, a_A, B_A, alpha_A, alpha_D 


def compute_screening(params):
    """ 
        Computes kappa_D : the inverse "Debye lenght assocaited with the 
        monovalent ion only
    """

    c_Na = params["c_Na"] # concentraion in unit of M 
    c_Cl = params["c_Cl"]

    charge_sum = (
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )
    # kappa_D only involved monovalent salt !!
    charge_sum *= (NA * 1000) # conversion  M=mol/l 

    kappa_D = np.sqrt(charge_sum / (epsilon * kB * T))
   
    return kappa_D

def Lambda_k(k, epsilon, kappa):
    return (k**2) / (epsilon * (k**2 + kappa**2))



def build_Gamma(k, params):
    
    D = np.array([
        params["D_M"],
        params["D_A"],
        params["D_MA"],
        params["D_D"],
        params["D_MB"]
    ])
    
    q = np.array([
        params["qM"],
        params["qA"],
        params["qMA"],
        params["qD"],
        params["qMB"]
    ])
    
    c_M =  params["c_M"]
    c_D, c_MB, theta_D = solve_DNA_reaction(params)
    c_A, c_MA, theta_A = solve_ATP_reaction(params)

    # print([c_M,c_A,c_MA,c_D,c_MB])

    c0 = np.array([c_M,c_A,c_MA,c_D,c_MB])

    c0*=c0 * NA *1000 # conversion 
    
    kBT = kB*T
    kappa = compute_screening(params)
    
    mu = D / kBT
    
    Lambda = Lambda_k(k, epsilon, kappa)
    
    N = 5
    Gamma = np.zeros((N, N))
    
    # diffusion (diagonal)
    for i in range(N):
        Gamma[i, i] = D[i] * k**2
    
    # electrostatic coupling (full matrix)
    for i in range(N):
        for j in range(N):
            Gamma[i, j] += mu[i] * q[i] * c0[i] * Lambda * q[j]
    
    return Gamma

def build_R(params):
    
    aD, BD, aA, BA, alphaA, alphaD = compute_coefficients(params)
    
    R = np.zeros((5,5))
    
    # order: (M, A, MA, D, MB)
    
    # Mg
    R[0] = [-(aA + aD), -alphaA, BA, -alphaD, BD]
    
    # A
    R[1] = [-aA, -alphaA, BA, 0, 0]
    
    # MA
    R[2] = [aA, alphaA, -BA, 0, 0]
    
    # D
    R[3] = [-aD, 0, 0, -alphaD, BD]
    
    # MB
    R[4] = [aD, 0, 0, alphaD, -BD]
    
    return R

def build_J_5x5(k, params):
    
    Gamma = build_Gamma(k, params)
    R = build_R(params)
    
    J = -Gamma + R
    
    return J

def compute_eigenmodes(k, params):
    
    J = build_J_5x5(k, params)
    
    eigvals, eigvecs = np.linalg.eig(J)
    
    return eigvals, eigvecs

if __name__ == "__main__":

    L = np.linspace(10e-9,1000e-9,10)


    for l in L:

        k = 2*np.pi/l
    
        eigvals, eigvecs = compute_eigenmodes(k, params)

        #print("Eigenvalues:")
        print("k=",k, eigvals)
        # print(np.real(eigvals))

   
