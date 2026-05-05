import numpy as np
import matplotlib.pyplot as plt


extforce='electric'

delta_phi_ext = 0.0000000001

#---------------------------------------------------------
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
    # diffusion constant (m^2/s)
    "D_M":1e-11,

    # reaction rates 
    "k_on_D":1e7,           #  in 1/(Ms) 
    "k_off_D":1e4,          #  in 1/s
    "k_on_A": 1.44e8,       #  in 1/(Ms) 
    "k_off_A": 7e3,         #  in 1/s

    # concentration in M
    "c_M": 1e-3,
    "c_ATP":4e-3,
    "S_0":50e-3,
    "c_Na":0.15,
    "c_Cl":0.15,

    # charges 
    "q_M":-2*e,
    "q_A":-4*e,
    "q_MA":-2*e,
    "q_D":-2*e,
    "q_MB":0

}



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
    c_D = theta_D * S_0
    c_MB  = (1-theta_D) * S_0
    
    return c_D, c_MB, theta_D

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
    k_on_D  = params["k_on_D"]
    k_off_D = params["k_off_D"]
    k_on_A  = params["k_on_A"]
    k_off_A = params["k_off_A"]

    c_M = params["c_M"]
    c_ATP = params["c_ATP"]

    c_D, _, _ = solve_reaction_DNA(params)
    c_A, _, _ = solve_reaction_ATP(params)

    a_D = k_on_D * c_D
    B_D = k_on_D * c_M + k_off_D
    a_A = k_on_A * c_A
    B_A = k_on_A * c_M + k_off_A 

    return a_D, B_D, a_A, B_A

# ---------------------------------------------------------
# Debye screening
# ---------------------------------------------------------


def compute_screening(params):
    c_M = params["c_M"]
    c_Na = params["c_Na"]
    c_Cl = params["c_Cl"]
    
    qM, qA, qD, qMA, qMB  = params["q_M"], params["q_A"], params["q_D"], params["q_MA"], params["q_MB"]

    deltaqA = qMB​-qD
    deltaqA = qMA​-qA

    # Kappa_D only involved monovalent salt !!
    charge_sum = (
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )

    charge_sum *= (NA * 1000) # conversion  M=mol/l 

    kappa_D = np.sqrt(charge_sum / (epsilon * kB * T))
    kappa_M2 = (qM)**2 * c_M * 1000 * NA / (epsilon * kB * T)
    kappa_A2 = (qM * deltaqA ) * c_M * 1000 * NA / (epsilon * kB * T)
    kappa_D2 = (qM * deltaqD ) * c_M * 1000 * NA / (epsilon * kB * T)


    return kappa_D, kappa_M2, kappa_A2, kappa_D2



def Gamma_ME(k, params):
    D_M = params["D_M"]
    kappa_D, kappa_M2, _ , _ = compute_screening(params)
    Gamma_ME = D_M * kappa_M2 * (k**2 / (k**2 + kappa_D**2))

    # return D_M * k**2 + Gamma_E
    return  Gamma_E

def Gamma_AE(k, params):
    D_M = params["D_M"]
    kappa_D, kappa_M2, kappa_A2,_ = compute_screening(params)
    Gamma_E = D_M * kappa_A2 * (k**2 / (k**2 + kappa_D**2))
    return + Gamma_E

def Gamma_DE(k, params):
    D_M = params["D_M"]
    kappa_D, kappa_M2, kappa_A2, kappa_D2 = compute_screening(params)
    Gamma_E = D_M * kappa_D2 * (k**2 / (k**2 + kappa_D**2))
    return + Gamma_E

def Gamma_M(k, params):
    D_M = params["D_M"]
    Gamma_E = Gamma_ME(k, params)
    return D_M * k**2 + Gamma_E


# ---------------------------------------------------------
#  applied-field forcing
# ---------------------------------------------------------

def external_force(k, params):
    match extforce:
        case 'driven':
            S = np.array([-1.0, -0.1, -0.1])
        case 'electric':
            # electric-field-driven forcing)

            q_M = 2*e
            mu_M = params["D_M"] / (kB * T)

            delta_phi = delta_phi_ext # arbitrary normalization

            S_M = - mu_M * q_M * params["c_M"] * k**2 * delta_phi

            S = np.array([S_M, 0.0, 0.0])

    return S 


# ---------------------------------------------------------
# Mg release (DNA vs ATP)
# ---------------------------------------------------------

def compute_release_components(freq, k, params):

    a_D, B_D, a_A, B_A = compute_coefficients(params)
   
    omega = 2 * np.pi * freq
    Gamma = Gamma_M(k, params)
    GammaA = Gamma_AE(k, params)
    GammaD = Gamma_DE(k, params)

    # Jacobian
    J = np.array([
        [-(Gamma + a_D + a_A), B_D+GammaA, B_A+GammaD],
        [a_D, -B_D, 0],
        [a_A, 0, -B_A]
    ])

    eigvals, V = np.linalg.eig(J)
    W = np.linalg.inv(V)

    S = external_force(k, params)

    R_total = np.zeros_like(freq, dtype=complex)
    R_DNA = np.zeros_like(freq, dtype=complex)
    R_ATP = np.zeros_like(freq, dtype=complex)

    for i in range(3):

        lambda_i = eigvals[i]
        v_i = V[:, i]
        w_i = W[i, :]

        A_i = np.dot(w_i, S)

        v_M, v_B, v_A = v_i

        C_D = B_D * v_B - a_D * v_M
        C_A = B_A * v_A - a_A * v_M

        R_total += (A_i * (C_D + C_A)) / (1j*omega - lambda_i)
        R_DNA += (A_i * C_D) / (1j*omega - lambda_i)
        R_ATP += (A_i * C_A) / (1j*omega - lambda_i)

    return R_total, R_DNA, R_ATP


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_release(k, params):

    freq = np.logspace(0, 5, 1000)

    R_total, R_DNA, R_ATP = compute_release_components(freq, k, params)

    print("check freq->0 R_ATP -> 0 R_DNA -> 0")
    print(freq[0], R_ATP[0],R_DNA[0])

    # ---- amplitude
    plt.figure()
    plt.loglog(freq, np.abs(R_total), label="Total")
    plt.loglog(freq, np.abs(R_DNA), label="DNA")
    plt.loglog(freq, np.abs(R_ATP), label="ATP")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude |R|")
    plt.title("Mg release rate amplitude")
    plt.legend()
    plt.grid(True)

    # ---- real part (more physical)
    plt.figure()
    plt.semilogx(freq, np.real(R_total), label="Total")
    plt.semilogx(freq, np.real(R_DNA), label="DNA")
    plt.semilogx(freq, np.real(R_ATP), label="ATP")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Re(R)")
    plt.title("Mg release rate (real part)")
    plt.legend()
    plt.grid(True)

    plt.show()



# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    R = 10e-9
    k = 2*np.pi / R


    plot_release(k, params)