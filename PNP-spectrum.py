import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Adjustable physical parameters
# ---------------------------------------------------------

params = {

    # diffusion
    "D_M":1e-10,        # Mg diffusion coefficient (m^2/s)

    # Mg-DNA kinetics
    "k_on_D":1e7,#1e7,       # M^-1 s^-1
    "k_off_D":1e4,#5e4      # s^-1

    # Mg-ATP kinetics
    "k_on_A":1.44e8,
    "k_off_A":7e3,

    # concentrations
    "c_M":1e-3,
    "c_ATP":4e-3,
    "S_0":50e-3,

    # background ions
    "c_Na":0.15,
    "c_Cl":0.15
}

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

kB = 1.380649e-23
T = 298
e = 1.602e-19
NA = 6.022e23
eps0 = 8.854e-12
eps_r = 78

epsilon = eps_r * eps0


# ---------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------

def solve_reaction(params):
     
    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    S_0 = params["S_0"]
    c_M = params["c_M"]
    
    K_D= k_off_D/k_on_D
    
    theta = 1.0/(1.0 + K_D/c_M) 
    S_free = (1.0 -theta ) * S_0

    return S_free, theta

def compute_coefficients(params):

    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]

    k_on_A = params["k_on_A"]
    k_off_A = params["k_off_A"]

    c_M = params["c_M"]
    c_ATP = params["c_ATP"]
    S_0 = params["S_0"]

    # fraction of bound sites 


    S_free, theta = solve_reaction(params)

    a_D = k_on_D * S_free
    B_D = k_on_D * c_M + k_off_D

    a_A = k_on_A * c_ATP
    B_A = k_off_A

    return a_D, B_D, a_A, B_A


# ---------------------------------------------------------
# Debye screening
# ---------------------------------------------------------

def compute_screening(params):

    c_M = params["c_M"]
    c_Na = params["c_Na"]
    c_Cl = params["c_Cl"]

    charge_sum = (
        (2*e)**2 * c_M +
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    ) 

    charge_sum *= (NA* 1000.0) 

    kappa_D = np.sqrt(charge_sum/(epsilon*kB*T))

    kappa_M2 = (2*e)**2 * c_M * 1000* NA /(epsilon*kB*T)

    return kappa_D, kappa_M2


# ---------------------------------------------------------
# transport operator
# ---------------------------------------------------------

def Gamma_M(k,params):

    D_M = params["D_M"]

    kappa_D,kappa_M2 = compute_screening(params)

    Gamma_E = D_M * kappa_M2 * (k**2/(k**2 + kappa_D**2))

    return D_M*k**2 + Gamma_E


# ---------------------------------------------------------
# Admittance function
# ---------------------------------------------------------

def compute_admittance(freq,k,params):

    a_D,B_D,a_A,B_A = compute_coefficients(params)

    omega = 2*np.pi*freq
    iomega = 1j*omega

    Gamma = Gamma_M(k,params)

    den = (
        iomega
        + Gamma
        + a_D * (iomega/(iomega+B_D))
        + a_A * (iomega/(iomega+B_A))
    )

    Y = iomega/den

    return Y


# ---------------------------------------------------------
# Plot admittance / conductance / capacitance
# ---------------------------------------------------------

def plot_response(k,params):

    freq = np.logspace(0,6,2000)

    Y = compute_admittance(freq,k,params)

    G = np.real(Y)
    C = np.imag(Y)/(2*np.pi*freq)

    # --- admittance components
    plt.figure()
    plt.semilogx(freq,np.real(Y),label="Re(Y)")
    plt.semilogx(freq,np.imag(Y),label="Im(Y)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Admittance")
    plt.title("Admittance components")
    plt.legend()

    # --- conductance
    plt.figure()
    plt.loglog(freq,G)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Conductance G")
    plt.title("Conductance vs frequency")

    # --- capacitance
    plt.figure()
    plt.loglog(freq,C)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Capacitance C")
    plt.title("Capacitance vs frequency")

    plt.show()


# ---------------------------------------------------------
# Slow frequency vs loop radius
# ---------------------------------------------------------

def plot_loop_dependence(params):

    plt.figure()

    D_M=[1e-11,1e-10,1e-9]

    a_D,B_D,a_A,B_A = compute_coefficients(params)

    for DifM in D_M:

        params["D_M"]=DifM

        R = np.linspace(20e-9,500e-9,300)

        freq_slow=[]

        for r in R:

            k = 2*np.pi/r
            Gamma = Gamma_M(k,params)

            J = np.array([
                [-(Gamma+a_D+a_A),B_D,B_A],
                [a_D,-B_D,0],
                [a_A,0,-B_A]
            ])

            eigvals=np.linalg.eigvals(J)

            slow=np.min(np.abs(eigvals))


            freq_slow.append(slow/(2*np.pi))

        
        plt.plot(R*1e9,freq_slow,label=f"Dm {params["D_M"]}")
        plt.xlabel("Loop radius (nm)")
        plt.ylabel("Slow frequency (Hz)")
        plt.title("Slow relaxation vs loop size")
        plt.legend()
    plt.show()

 
def test_freq(params):

    print("Derived Parameters:")
    
    S_free, theta = solve_reaction(params)
    print("S_free  =", S_free)
    print("theta   =", theta)

    a_D,B_D,a_A,B_A = compute_coefficients(params)
   
    print("a_D     =", a_D)
    print("a_A     =", a_A)
    print("B_D     =", B_D)
    print("B_A     =", B_A) 
   
    k = 1e7
    Gamma = Gamma_M(k,params)

    print("Gamma_M =", Gamma)

    J = np.array([
        [-(Gamma+a_D+a_A),B_D,B_A],
        [a_D,-B_D,0],
        [a_A,0,-B_A]        ])

    eigvals=np.sort(np.linalg.eigvals(J))[::-1]

    freq_exact = np.abs(eigvals)/(2.0*np.pi)

    print("\nExact eigenvalues (s^-1)")
    print(eigvals)

    print("\nExact frequencies (Hz)")
    print(freq_exact)    


# ---------------------------------------------------------
# Run example
# ---------------------------------------------------------

if __name__ == "__main__":

    # choose loop radius
    # R = 50e-9
    # k = 1/R

    # plot_response(k,params)

    plot_loop_dependence(params)
    #test_freq(params)