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
    # diffusion constants  coefficient (m^2/s)
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
    """ 
        Returns : a_D, B_D, a_A, B_A, alpha_A 
    """

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
    GMA_MA = D_MA*k**2 + Gamma(muMA, qMA, cMA0, qMA)

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

    #eigvals, eigvecs = eig(J)
    #idx = np.argmax(np.real(eigvals))
    #slow_vec = eigvecs[:, idx]

    #print("Slow mode composition:", slow_vec)

    return Deff, eigvals, lam_slow


def analytic_Deff_M(k, params):
    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    D_M = params["D_M"]
    qM = params["qM"]

    cM0 = params["c_M"] * NA * 1000

    kBT = kB * T
    kappa = compute_screening(params)

    # normalization
    ZM = 1 + aD/BD + aA/(BA + alphaA)

    # effective charge
    qA = params["qA"]
    qMA = params["qMA"]

    q_eff = qM + (qMA - qA) * aA/(BA + alphaA)

    return (D_M / ZM) * (
        1 + (qM * q_eff * cM0)/(epsilon * kBT * (k**2 + kappa**2))
    )


def analytic_Deff_A(k, params):
    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    D_A = params["D_A"]
    qA = params["qA"]

    cA0 = solve_ATP_reaction(params)[0] * NA * 1000

    kBT = kB * T
    kappa = compute_screening(params)

    # normalization
    thetaD = aD/BD
    thetaA = aA/BA
    etaA = alphaA/BA

    ZA = 1 + etaA * (1 + thetaD)/(1 + thetaA + thetaD)

    qM = params["qM"]
    qMA = params["qMA"]

    q_eff = (
        qA
        + qMA * etaA*(1+thetaD)/(1+thetaA+thetaD)
        - qM * etaA/(1+thetaA+thetaD)
    )

    return (D_A / ZA) * (
        1 + (qA * q_eff * cA0)/(epsilon * kBT * (k**2 + kappa**2))
    )

def analytic_Deff_M_with_Zm_q_eff(k, params):
    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    D_M = params["D_M"]
    qM = params["qM"]

    cM0 = params["c_M"] * NA * 1000

    kBT = kB * T
    kappa = compute_screening(params)

    # normalization
    ZM = 1 + aD/BD + aA/(BA + alphaA)

    # effective charge
    qA = params["qA"]
    qMA = params["qMA"]

    q_eff = qM + (qMA - qA) * aA/(BA + alphaA)

    D_eff=(D_M / ZM) * (
        1 + (qM * q_eff * cM0)/(epsilon * kBT * (k**2 + kappa**2))
    )
    EM= (qM * q_eff * cM0)/(epsilon * kBT * (k**2 + kappa**2))
    return(D_eff,Z_M,q_eff,EM)


def analytic_Deff_corrected(k, params):

    # --- coefficients ---
    aD, BD, aA, BA, alphaA = compute_coefficients(params)

    D_M = params["D_M"]
    D_MA = params["D_MA"]

    qM = params["qM"]
    qA = params["qA"]
    qMA = params["qMA"]

    kBT = kB * T
    kappa = compute_screening(params)

    # concentrations (number density)
    cM0 = params["c_M"] * NA * 1000
    cA0, cMA0, _ = solve_ATP_reaction(params)
    cMA0 *= NA * 1000

    # ---- effective charge (same as before) ----
    q_eff = qM + (qMA - qA) * aA / (BA + alphaA)

    # ---- electrostatic term for Mg ----
    D_el = D_M * (qM * q_eff * cM0) / (epsilon * kBT * (k**2 + kappa**2))

    # ---- Gamma_MA(k) (transport of complex) ----
    Gamma_MA = (
        D_MA * k**2
        +
        D_MA * (qMA**2 * cMA0) *(k**2) / (epsilon * kBT * (k**2 + kappa**2))
    )

    # ---- kinetic rate ----
    Lambda_A = BA + alphaA

    # ---- kinetic correction factor ----
    Fk = Gamma_MA / (Gamma_MA + Lambda_A)

    # ---- trapping factors ----
    theta_D = aD / BD
    theta_A = aA / (BA + alphaA)
    ZM = 1 + theta_D + theta_A * Fk

    # ---- final corrected Deff ----
    Deff = (D_M + D_el + Fk * aA  / k**2 ) / ZM

    print(Deff,D_M,D_el,Fk,Lambda_A,BA)

    return Deff,Fk



def compute_release_components(freq, k, params):

    D_M, D_A, D_MA = params["D_M"], params["D_A"], params["D_MA"]
    a_D, B_D, a_A, B_A, alpha_A = compute_coefficients(params)
    qM, qA, qMA = params["qM"], params["qA"], params["qMA"]

    kBT = kB * T

    omega = 2 * np.pi * freq

    # concentrations  number density
    cM0 = params["c_M"]
    cA0, cMA0, _ = solve_ATP_reaction(params)
   

    J = build_J(k, params)
    eigvals, eigvecs = eig(J)
    w = np.linalg.inv(eigvecs)


    u_A = np.array([-alpha_A, -a_A, B_A, 0])
    u_D = np.array([-a_D, 0, 0, B_D])

    S = -k**2 * delta_phi_ext * np.array([
        D_M*qM*cM0/kBT,
        D_A*qA*cA0/kBT,
        D_MA*qMA*cMA0/kBT,
        0
    ])

    R_ATP = np.zeros_like(freq, dtype=complex)
    R_DNA = np.zeros_like(freq, dtype=complex)

    for i in range(4):
        vi = eigvecs[:, i]
        wi = w[i, :]

        Ai = np.dot(wi, S)

        contrib = Ai / (1j*omega - eigvals[i])

        R_ATP += contrib * np.dot(u_A, vi)
        R_DNA += contrib * np.dot(u_D, vi)

    R_total = R_ATP + R_DNA
    #
    print("---- Numeric check diagnostics ----")
    for i in range(4):
        for j in range(4):
            print(i, j, np.dot(w[i,:], eigvecs[:,j]))

    return R_total, R_DNA, R_ATP


#---------------------------------------------------------
# change of delta cM delta cMB and delta cMA
# ---------------------------------------------------------

def compute_delta_conc_components(freq, k, params):

    D_M, D_A, D_MA = params["D_M"], params["D_A"], params["D_MA"]
    qM, qA, qMA = params["qM"], params["qA"], params["qMA"]
    
    omega = 2 * np.pi * freq
    
    J = build_J(k, params)

    eigvals, V = np.linalg.eig(J)
    W = np.linalg.inv(V)
    
    # concentrations number density
    cM0 = params["c_M"]
    cA0, cMA0, _ = solve_ATP_reaction(params)

    kBT = kB * T

    S = -k**2 * delta_phi_ext * np.array([
        D_M*qM*cM0/kBT,
        D_A*qA*cA0/kBT,
        D_MA*qMA*cMA0/kBT,
        0
    ])

    
    delta_cM   = np.zeros_like(freq, dtype=complex)
    delta_cMB  = np.zeros_like(freq, dtype=complex)
    delta_cMA  = np.zeros_like(freq, dtype=complex)
    delta_cA   = np.zeros_like(freq, dtype=complex)

    for i in range(3):

        lambda_i = eigvals[i]
        v_i = V[:, i]
        w_i = W[i, :]

        A_i = np.dot(w_i, S)

        v_M, v_A,  v_MA, v_MB = v_i

        delta_cM +=  (A_i * v_M) / (1j*omega - lambda_i)
        delta_cA +=  (A_i * v_A) / (1j*omega - lambda_i)
        delta_cMA += (A_i * v_MA) / (1j*omega - lambda_i)
        delta_cMB += (A_i * v_MB) / (1j*omega - lambda_i)
    
    return delta_cM, delta_cA, delta_cMA, delta_cMB




# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

def plot_Deff_compare_corr(params):

    k_vals = np.logspace(5, 8, 100)

    Deff_num_vals = []
    Deff_M_vals = []
    Deff_corr_vals = []

    for k in k_vals:
        Deff_num, _, _ = compute_Deff(k, params)
        Deff_num_vals.append(Deff_num)

        Deff_M_vals.append(analytic_Deff_M(k, params))
        Deff_corr_vals.append(analytic_Deff_corrected(k, params)[0])

    plt.figure()
    plt.loglog(k_vals, Deff_num_vals, label="Numerical", linewidth=3)
    plt.loglog(k_vals, Deff_M_vals, "--", label="Mg-limit (equilibrium)")
    plt.loglog(k_vals, Deff_corr_vals, "-.", label="Corrected (finite kinetics)")

    plt.xlabel("k")
    plt.ylabel("D_eff")
    plt.legend()
    plt.title("Numerical vs analytic (corrected) D_eff")
    plt.show()

    Fk_vals = [analytic_Deff_corrected(k, params)[1] for k in k_vals]

    plt.figure()
    plt.semilogx(k_vals, Fk_vals)
    plt.xlabel("k")
    plt.ylabel("F(k)")
    plt.title("Kinetic crossover function")
    plt.show()


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



def plot_Deff(params):

    k_vals = np.logspace(5, 8, 50)
    Deff_vals = []

    for k in k_vals:
        Deff, _ , _  = compute_Deff(k, params)
        Deff_vals.append(Deff)
        #automatic regime classification
        if Deff < params["D_A"] * 0.1:
            regime = "Mg-controlled"
        elif Deff > params["D_A"] * 0.5:
            regime = "ATP-controlled"
        else:
            regime = "Coupled"
        print(k,Deff,regime)

    plt.figure()
    plt.loglog(k_vals, Deff_vals)
    plt.xlabel("k")
    plt.ylabel("D_eff")
    plt.title("Effective diffusion")
    plt.show()


def plot_loop_dependence(params):

    plt.figure()

    D_M=[1e-11,1e-10,1e-9]
    D_A=[1e-14,1e-13,1e-12,1e-11,1e-10]

    D_M=[params["D_M"]]
    D_A=[params["D_A"]]

    for DifM in D_M:
        for DifA in D_A:

            params["D_M"]=DifM
            params["D_A"]=DifA

            R = np.linspace(20e-9,500e-9,300)
            freq_slow=[]

            for r in R:
                k = 2*np.pi/r
                Deff, eigvals, lam_slow  = compute_Deff(k, params)
                freq_slow.append(np.abs(lam_slow)/(2*np.pi))

            
            plt.plot(R*1e9,freq_slow,label=f"Dm {params["D_M"]} Da {params["D_A"]}")
            plt.xlabel("Loop radius (nm)")
            plt.ylabel("Slow frequency (Hz)")
            plt.title("Slow relaxation vs loop size")
            plt.legend()
    plt.show()


 
def plot_loop_dependence_default(params):

    plt.figure()
    
    R = np.linspace(20e-9,500e-9,300)
    freq_slow=[]

    for r in R:
        k = 2*np.pi/r
        Deff, eigvals, lam_slow  = compute_Deff(k, params)
        freq_slow.append(np.abs(lam_slow)/(2*np.pi))

            
    plt.plot(R*1e9,freq_slow,label=f"Dm {params["D_M"]} Da {params["D_A"]}")
    plt.xlabel("Loop radius (nm)")
    plt.ylabel("Slow frequency (Hz)")
    plt.title("Slow relaxation vs loop size")
    plt.legend()
    plt.show()
  


def plot_Deff_compare(params):

    k_vals = np.logspace(5, 8, 50)
    Deff_num_vals = []
    Deff_M_vals = []
    Deff_A_vals = []

    for k in k_vals:
        Deff_num, _ , _= compute_Deff(k, params)
        Deff_num_vals.append(Deff_num)

        Deff_M_vals.append(analytic_Deff_M(k, params))
        Deff_A_vals.append(analytic_Deff_A(k, params))


    plt.figure()
    plt.loglog(k_vals, Deff_num_vals, label="Numerical", linewidth=3)
    plt.loglog(k_vals, Deff_M_vals, "--", label="Mg-limit")
    plt.loglog(k_vals, Deff_A_vals, "--", label="ATP-limit")

    plt.xlabel("k")
    plt.ylabel("D_eff")
    plt.legend()
    plt.title("Numerical vs analytic D_eff")
    plt.show()



def plot_release(k, params):

    freq = np.logspace(0, 5, 1000)

    R_total, R_DNA, R_ATP = compute_release_components(freq, k, params)

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


def plot_delta_conc(k, params):

    freq = np.logspace(0, 5, 1000)

    delta_cM, delta_cA, delta_cMA, delta_cMB = compute_delta_conc_components(freq, k, params)

    # ---- amplitude
    plt.figure()
    plt.loglog(freq, np.abs(delta_cM), label="cM")
    plt.loglog(freq, np.abs(delta_cA), label="cA")
    plt.loglog(freq, np.abs(delta_cMA), label="cMA")
    plt.loglog(freq, np.abs(delta_cMB), label="cMB")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(" |delta c|")
    plt.title("delta concentration amplitude")
    plt.legend()

    # ---- real part (more physical)
    plt.figure()
    plt.semilogx(freq, np.real(delta_cM), label="cM")
    plt.semilogx(freq, np.real(delta_cA), label="cA")
    plt.semilogx(freq, np.real(delta_cMA), label="cMA")
    plt.semilogx(freq, np.real(delta_cMB), label="cMB")
   
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Re( delta c)")
    plt.title("delta concentration (real part)")
    plt.legend()

    plt.show()



if __name__ == "__main__":

    #phase_diagram(params)
    #plot_phase_diagram()

    #plot_Deff(params)
    # plot_loop_dependence(params)
    
   # D_M=[1e-11,1e-10,1e-9]


    #for DifM in D_M:
    #    params["D_M"]=DifM
    #    #plot_Deff_compare(params)
    #    plot_Deff_compare_corr(params)

    plot_loop_dependence_default(params)

    R = 50e-9
    k = 2*np.pi / R
    plot_release(k, params)
    plot_delta_conc(k, params)
   
