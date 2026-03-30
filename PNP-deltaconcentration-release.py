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
        (2*e)**2 * c_M +
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )

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
# change of delta cM delta cMB and delta cMA
# ---------------------------------------------------------

def compute_delta_conc_components(freq, k, params):

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
    
    delta_cM   = np.zeros_like(freq, dtype=complex)
    delta_cMB  = np.zeros_like(freq, dtype=complex)
    delta_cMA  = np.zeros_like(freq, dtype=complex)

    for i in range(3):

        lambda_i = eigvals[i]
        v_i = V[:, i]
        w_i = W[i, :]

        A_i = np.dot(w_i, S)

        v_M, v_B, v_A = v_i

        delta_cM +=  (A_i * v_M) / (1j*omega - lambda_i)
        delta_cMB += (A_i * v_B) / (1j*omega - lambda_i)
        delta_cMA += (A_i * v_A) / (1j*omega - lambda_i)

    return delta_cM, delta_cMB, delta_cMA 



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

    # ---- real part (more physical)
    plt.figure()
    plt.semilogx(freq, np.real(R_total), label="Total")
    plt.semilogx(freq, np.real(R_DNA), label="DNA")
    plt.semilogx(freq, np.real(R_ATP), label="ATP")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Re(R)")
    plt.title("Mg release rate (real part)")
    plt.legend()

    plt.show()


def plot_delta_conc(k, params):

    freq = np.logspace(0, 5, 1000)

    delta_cM, delta_cMB, delta_cMA = compute_delta_conc_components(freq, k, params)

    # ---- amplitude
    plt.figure()
    plt.loglog(freq, np.abs(delta_cM), label="cM")
    plt.loglog(freq, np.abs(delta_cMB), label="cMB")
    plt.loglog(freq, np.abs(delta_cMA), label="cMA")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(" |delta c|")
    plt.title("delta concentration amplitude")
    plt.legend()

    # ---- real part (more physical)
    plt.figure()
    plt.semilogx(freq, np.real(delta_cM), label="cM")
    plt.semilogx(freq, np.real(delta_cMB), label="cMB")
    plt.semilogx(freq, np.real(delta_cMA), label="cMA")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Re( delta c)")
    plt.title("delta concentration (real part)")
    plt.legend()

    plt.show()

# ---------------------------------------------------------
# Concentration responses + Mg release
# ---------------------------------------------------------

def compute_concentrations_and_release(freq, k, params):

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

    # electric-field forcing
    S = external_force(k, params)
   
    # initialize
    cM = np.zeros_like(freq, dtype=complex)
    cMB = np.zeros_like(freq, dtype=complex)
    cMA = np.zeros_like(freq, dtype=complex)

    R_total = np.zeros_like(freq, dtype=complex)

    for i in range(3):

        lam = eigvals[i]
        v = V[:, i]
        w = W[i, :]

        A = np.dot(w, S)

        v_M, v_B, v_A = v

        resp = A / (1j*omega - lam)

        # concentrations
        cM  += resp * v_M
        cMB += resp * v_B
        cMA += resp * v_A

        # release coefficients
        C_D = B_D*v_B - a_D*v_M
        C_A = B_A*v_A - a_A*v_M

        R_total += resp * (C_D + C_A)

    return cM, cMB, cMA, R_total

def plot_concentrations_and_release(k, params):

    freq = np.logspace(0, 5, 1000)

    cM, cMB, cMA, R = compute_concentrations_and_release(freq, k, params)

    # --- amplitudes
    plt.figure()
    plt.loglog(freq, np.abs(cM), label="|δc_M| (free Mg)")
    plt.loglog(freq, np.abs(cMB), label="|δc_MB| (DNA)")
    plt.loglog(freq, np.abs(cMA), label="|δc_MA| (ATP)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Concentration response amplitudes")
    plt.legend()
    plt.grid(True)

    # --- real parts (important!)
    plt.figure()
    plt.semilogx(freq, np.real(cM), label="Re(δc_M)")
    plt.semilogx(freq, np.real(cMB), label="Re(δc_MB)")
    plt.semilogx(freq, np.real(cMA), label="Re(δc_MA)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Real part")
    plt.title("In-phase concentration response")
    plt.legend()
    plt.grid(True)

    # --- Mg release
    plt.figure()
    plt.semilogx(freq, np.real(R), label="Re(R)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mg release")
    plt.title("Mg release vs frequency")
    plt.legend()
    plt.grid(True)

    plt.show()


# ---------------------------------------------------------
# Equilibrium deviation + phase difference + unified plot
# ---------------------------------------------------------

def compute_deviation_phase_release(freq, k, params):

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

    # electric-field forcing
    
    S = external_force(k, params)

    # initialize
    cM  = np.zeros_like(freq, dtype=complex)
    cMB = np.zeros_like(freq, dtype=complex)
    cMA = np.zeros_like(freq, dtype=complex)
    R   = np.zeros_like(freq, dtype=complex)

    for i in range(3):

        lam = eigvals[i]
        v = V[:, i]
        w = W[i, :]

        A = np.dot(w, S)
        resp = A / (1j*omega - lam)

        v_M, v_B, v_A = v

        # concentrations
        cM  += resp * v_M
        cMB += resp * v_B
        cMA += resp * v_A

        # release
        C_D = B_D * v_B - a_D * v_M
        C_A = B_A * v_A - a_A * v_M
        R  += resp * (C_D + C_A)

    # --- equilibrium deviations
    dev_ATP = cMA - (a_A / B_A) * cM
    dev_DNA = cMB - (a_D / B_D) * cM

    # --- phase differences
    phase_ATP = np.angle(cMA) - np.angle(cM)
    phase_DNA = np.angle(cMB) - np.angle(cM)

    return dev_ATP, dev_DNA, phase_ATP, phase_DNA, R


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_deviation_phase_unified(k, params):

    freq = np.logspace(0, 5, 1000)

    dev_ATP, dev_DNA, phase_ATP, phase_DNA, R = \
        compute_deviation_phase_release(freq, k, params)

    # -------------------------
    # 1. Deviation plot
    # -------------------------
    plt.figure()
    plt.loglog(freq, np.abs(dev_ATP), label="ATP deviation")
    plt.loglog(freq, np.abs(dev_DNA), label="DNA deviation")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Deviation magnitude")
    plt.title("Deviation from binding equilibrium")
    plt.legend()
    plt.grid(True)

    # -------------------------
    # 2. Phase difference plot
    # -------------------------
    plt.figure()
    plt.semilogx(freq, phase_ATP, label="ATP phase diff")
    plt.semilogx(freq, phase_DNA, label="DNA phase diff")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase difference (rad)")
    plt.title("Phase difference vs free Mg")
    plt.legend()
    plt.grid(True)

    # -------------------------
    # 3. Unified plot
    # -------------------------
    R_norm   = np.abs(R) / np.max(np.abs(R))
    dev_norm = np.abs(dev_ATP) / np.max(np.abs(dev_ATP))
    phase_norm = np.abs(phase_ATP) / np.max(np.abs(phase_ATP))

    plt.figure()
    plt.loglog(freq, R_norm, label="Release (norm)")
    plt.loglog(freq, dev_norm, label="Equilibrium deviation (norm)")
    plt.loglog(freq, phase_norm, label="Phase difference (norm)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized magnitude")
    plt.title("Unified: Release vs deviation vs phase")
    plt.legend()
    plt.grid(True)

    plt.show()


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    R = 50e-9
    k = 2*np.pi / R

    plot_deviation_phase_unified(k, params)

    #plot_release(k, params)
    
    plot_concentrations_and_release(k, params)
    
