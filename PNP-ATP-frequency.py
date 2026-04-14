import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

extforce='electric'

delta_phi_ext = 0.0000000001

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
    # diffusion constants 
    "D_M":1e-11,
    "D_A":3e-10,
    "D_MA":3e-10,
    # k_on and k_off rates 
    "k_on_D":1e7,           #  in 1/(Ms) 
    "k_off_D":1e4,          #  in 1/s
    "k_on_A": 1.44e8,       #  in 1/(Ms) 
    "k_off_A": 7e3,         #  in 1/s
    # concentrations in M
    "S_0":50e-3,  # total concentration DNA 
    "c_M": 1e-3,
    "c_ATP":4e-3, # total c_ATP = c_A + c_MA 
    "c_Na":0.15,
    "c_Cl":0.15,
    # charges in C
    "qM":2*e,
    "qA":-4*e,
    "qMA":-2*e
}

print(params)


# ---------------------------------------------------------
# Reaction + transport
# ---------------------------------------------------------

def solve_DNA_reaction(params):
    """ 
        Returns : S_free and theta
        S_free: concentration of phosphates that have not Mg bound 
        theta_D : fraction of phospahte that is not bound with Mg
    """
    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    S_0 = params["S_0"]
    c_M = params["c_M"]

    K_D = k_off_D / k_on_D
    
    #theta = 1.0 / (1.0 + K_D / c_M) =1 -theta_D
    #S_free = (1.0 - theta) * S_0 

    theta_D = 1.0 / (1.0 + c_M/K_D)
    S_free = theta_D * S_0
    
    return S_free, theta_D


def solve_ATP_reaction(params):
    """ 
        Returns : c_A, c_MA, theta_A
        c_free: concentration of ATP that has not Mg bound 
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

    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    k_on_A = params["k_on_A"]
    k_off_A = params["k_off_A"]
    c_M = params["c_M"]

    S_free, _ =  solve_DNA_reaction(params)
    c_A, _ , _= solve_ATP_reaction(params)


    a_D = k_on_D * S_free
    B_D = k_on_D * c_M + k_off_D
    a_A = k_on_A * c_A
    B_A = k_off_A
    alpha_A = k_on_A * c_M

    return a_D, B_D, a_A, B_A, alpha_A


def compute_screening(params):
    c_Na = params["c_Na"]
    c_Cl = params["c_Cl"]

    charge_sum = (
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )
    # kappa_D only involved monovalent salt !!
    charge_sum *= (NA * 1000) # conversion  M=mol/l 

    kappa_D = np.sqrt(charge_sum / (epsilon * kB * T))
   
    return kappa_D



# ---------------------------------------------------------
# Jacobian with Dynamic ATP 
# ---------------------------------------------------------

def build_J(k, params):
    # Unpack parameters
    D_M, D_A, D_MA = params["D_M"], params["D_A"], params["D_MA"]
    cM0 = params["c_M"]
    qM, qA, qMA = params["qM"], params["qA"], params["qMA"]

    kBT = kB * T 
    #eps = params["eps"] +> epsiol
    kappa = compute_screening(params)

    #aD, BD = params["aD"], params["BD"]
    #aA, BA = params["aA"], params["BA"]
    #alphaA = params["alphaA"]

    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    # Electrostatic kernel
    Lambda = k**2 / (epsilon * (k**2 + kappa**2))

    # Mobility
    muM = D_M / kBT
    muA = D_A / kBT
    muMA = D_MA / kBT

    # free ATP and bound Mg with ATP
    cA0, cMA0, thetaA = solve_ATP_reaction(params)
   
    # print("free ATP=",cA0," bound ATP",cMA0, (aA / BA) * cM0), " fraction=", thetaA

    # Electrostatic couplings Gamma_ij
    def Gamma(mu_i, qi, ci0, qj):
        return mu_i * qi * ci0 * NA * 1000 * Lambda * qj

    # Build Gamma matrices
    GM_M = D_M*k**2 + Gamma(muM, qM, cM0, qM)
    GM_A = Gamma(muM, qM, cM0, qA)
    GM_MA = Gamma(muM, qM, cM0, qMA)

    GA_M = Gamma(muA, qA, cA0, qM)
    GA_A = D_A*k**2 + Gamma(muA, qA, cA0, qA)
    GA_MA = Gamma(muA, qA, cA0, qMA)

    GMA_M = Gamma(muMA, qMA, cMA0, qM)
    GMA_A = Gamma(muMA, qMA, cMA0, qA)
    GMA_MA = D_MA*k**2 + Gamma(muMA, qMA, cM0, qMA)

    # Jacobian (4×4)
    J = np.array([
        [-(GM_M + aD + aA),   -(GM_A + alphaA),   -(GM_MA - BA),     BD],
        [-(GA_M + aA),        -(GA_A + alphaA),   -(GA_MA - BA),     0],
        [ (aA - GMA_M),       (alphaA - GMA_A),   -(GMA_MA + BA),    0],
        [ aD,                 0,                  0,                 -BD]
    ])

    return J

def compute_Deff(k, params):
    J = build_J(k, params)
    eigvals = eig(J)[0]

    # Sort by real part (closest to zero = slowest)
    eigvals_sorted = sorted(eigvals, key=lambda x: np.real(x))

    lam_slow = eigvals_sorted[-1]  # least negative
    Deff = -np.real(lam_slow) / k**2

    return Deff, eigvals

def phase_diagram(params):

    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    D_M = params["D_M"]
    D_A = params["D_A"]

    # trapping factor
    T_M = 1 + aD/BD + aA/BA

    D_M_eff = D_M / T_M

    Pi = D_A / D_M_eff

    print("---- Phase diagnostics ----")
    print(f"T_M (trapping) = {T_M:.2e}")
    print(f"D_M_eff = {D_M_eff:.2e}")
    print(f"D_A = {D_A:.2e}")
    print(f"Pi = {Pi:.2e}")

    if Pi > 10:
        print("→ Mg-controlled regime")
    elif Pi < 0.1:
        print("→ ATP-controlled regime")
    else:
        print("→ Coupled regime")



def plot_phase_diagram():

    D_ratio = np.logspace(-2, 2, 100)
    T_M_vals = np.logspace(0, 3, 100)

    Pi = np.outer(D_ratio, T_M_vals)

    plt.figure()
    plt.contourf(T_M_vals, D_ratio, Pi, levels=50)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("T_M (trapping)")
    plt.ylabel("D_A / D_M")
    plt.title("Phase diagram")

    plt.colorbar(label="Pi = D_A / D_M_eff")
    plt.show()

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

  
    phase_diagram(params)
    plot_phase_diagram()

    k_vals = np.logspace(5, 8, 50)
    Deff_vals = []

    for k in k_vals:
        Deff, eigvals = compute_Deff(k, params)
        Deff_vals.append(Deff)
        print(k,Deff,eigvals)

        # plot_release(k, params)