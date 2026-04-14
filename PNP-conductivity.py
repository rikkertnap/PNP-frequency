import numpy as np
import matplotlib.pyplot as plt


extforce='electric'


# ---------------------------------------------------------
# Physical parameters (same as your setup)
# ---------------------------------------------------------

params = {
    "D_M":1e-11,
    "k_on_D":1e7,
    "k_off_D":1e4,
    "k_on_A": 1.44e8,
    "k_off_A": 7e3,
    "c_M":1e-3,
    "c_ATP":4e-3,
    "S_0":50e-3,
    "c_Na":0.15,
    "c_Cl":0.15
}

kB = 1.380649e-23
T = 298
e = 1.602e-19
NA = 6.022e23
eps0 = 8.854e-12
eps_r = 78
epsilon = eps_r * eps0

# ---------------------------------------------------------
# Reaction + transport
# ---------------------------------------------------------

def solve_reaction(params):
    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    S_0 = params["S_0"]
    c_M = params["c_M"]

    K_D = k_off_D / k_on_D
    theta = 1.0 / (1.0 + K_D / c_M)
    S_free = (1.0 - theta) * S_0

    return S_free


def compute_coefficients(params):
    k_on_D = params["k_on_D"]
    k_off_D = params["k_off_D"]
    k_on_A = params["k_on_A"]
    k_off_A = params["k_off_A"]

    c_M = params["c_M"]
    c_ATP = params["c_ATP"]

    S_free = solve_reaction(params)

    a_D = k_on_D * S_free
    B_D = k_on_D * c_M + k_off_D

    a_A = k_on_A * c_ATP
    B_A = k_off_A

    return a_D, B_D, a_A, B_A


def compute_screening(params):
    c_M = params["c_M"]
    c_Na = params["c_Na"]
    c_Cl = params["c_Cl"]

    charge_sum = (
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )
    # kappa_D only involved monovalent salt !!
    charge_sum *= (NA * 1000)

    kappa_D = np.sqrt(charge_sum / (epsilon * kB * T))
    kappa_M2 = (2*e)**2 * c_M * 1000 * NA / (epsilon * kB * T)

    return kappa_D, kappa_M2


def Gamma_M(k, params):
    D_M = params["D_M"]
    kappa_D, kappa_M2 = compute_screening(params)

    Gamma_E = D_M * kappa_M2 * (k**2 / (k**2 + kappa_D**2))

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

            delta_phi = 0.001  # arbitrary normalization

            S_M = - mu_M * q_M * params["c_M"] * k**2 * delta_phi

            S = np.array([S_M, 0.0, 0.0])

    return S 

# ---------------------------------------------------------
# Admittance and conductance vs frequency
# ---------------------------------------------------------

def compute_admittance(freq, k, params):

    a_D, B_D, a_A, B_A = compute_coefficients(params)
    omega = 2*np.pi*freq
    Gamma = Gamma_M(k, params)

    # Jacobian
    J = np.array([
        [-(Gamma + a_D + a_A), B_D, B_A],
        [a_D, -B_D, 0],
        [a_A, 0, -B_A]
    ])

    eigvals, V = np.linalg.eig(J)
    W = np.linalg.inv(V)

    # physical constants
    q_M = 2*e
    mu_M = params["D_M"]/(kB*T)

    # forcing amplitude (delta phi_ext = 1)
    S = external_force(k, params)

    # effective diffusion (includes electrostatics)
    kappa_D, kappa_M2 = compute_screening(params)
    D_eff = params["D_M"] * (1 + kappa_M2/(k**2 + kappa_D**2))

    # compute δc_M / δφ_ext
    response = np.zeros_like(freq, dtype=complex)

    for i in range(3):
        lam = eigvals[i]
        v = V[:, i]
        w = W[i, :]

        A = np.dot(w, S)
        v_M = v[0]

        response += (A * v_M) / (1j*omega - lam)

    # since delta_phi_ext = 1, this is already δc_M / δφ_ext
    dcM_dphi = response

    # admittance
    Y = -1j * k * mu_M * q_M * params["c_M"] \
        -1j * k * D_eff * dcM_dphi

    return Y



# ---------------------------------------------------------
# Mg release (DNA vs ATP)
# ---------------------------------------------------------

def compute_release_components(freq, k, params):

    a_D, B_D, a_A, B_A = compute_coefficients(params)
    omega = 2 * np.pi * freq
    Gamma = Gamma_M(k, params)

    # Jacobian
    J = np.array([
        [-(Gamma + a_D + a_A), B_D, B_A],
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

def plot_admittance_conductance(k, params):

    freq = np.logspace(0, 5, 1000)

    Y = compute_admittance(freq, k, params)
    R, R_DNA, R_ATP = compute_release_components(freq, k, params)


    G = np.real(Y)   # conductance
    B = np.imag(Y)   # susceptance

    # --- Admittance magnitude
    plt.figure()
    plt.loglog(freq, np.abs(Y))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Y(ω)|")
    plt.title("Admittance magnitude")
    plt.grid(True)

    # --- Conductance
    plt.figure()
    plt.semilogx(freq, G)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("G(ω) = Re(Y)")
    plt.title("Conductance vs frequency")
    plt.grid(True)

    # --- Susceptance
    plt.figure()
    plt.semilogx(freq, B)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("B(ω) = Im(Y)")
    plt.title("Susceptance vs frequency")
    plt.grid(True)

     # --- Conductance and Release
    plt.figure()
    plt.semilogx(freq, G/np.max(G), label="Conductance")
    plt.semilogx(freq, np.abs(R)/np.max(np.abs(R)), label="Mg release")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.title("Conductance and Release vs frequency")
    plt.grid(True)



    plt.show()


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    R = 50e-9
    k = 2*np.pi / R

    plot_admittance_conductance(k, params)
