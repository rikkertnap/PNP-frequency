import numpy as np

# ---------------------------------------------------------
# Physical constants
# ---------------------------------------------------------

kB = 1.380649e-23
T = 298
e = 1.602e-19
NA = 6.022e23
eps0 = 8.854e-12
eps_r = 78

epsilon = eps_r * eps0


# ---------------------------------------------------------
# Biochemical parameters
# ---------------------------------------------------------

# Mg-DNA 
k_on_D = 1e7     #   in 1/(Ms) 
k_off_D = 1e4     #  in 1/s

# Mg-ATP
k_on_A = 1.44e8   #  in 1/(Ms) 
k_off_A = 7e3     #  in 1/s 

# concentrations in M
c_M = 1e-3
c_ATP = 4e-3
S_0 = 50e-3

# monovalent ions in M 
c_Na = 0.15
c_Cl = 0.15

# diffusion in m^2/s
D_M = 1e-10

# wavenumber in 1/m
k = 1e7


# ---------------------------------------------------------
# Derived reaction coefficients
# --------------------------------------------------------- 

# fraction of bound sites 

def solve_reaction():
    K_D= k_off_D/k_on_D
    
    theta = 1.0/(1.0 + K_D/c_M) 
    S_free = (1.0 -theta ) * S_0

    return S_free, theta

S_free, theta = solve_reaction()

a_D = k_on_D * S_free
B_D = k_on_D * c_M + k_off_D

a_A = k_on_A * c_ATP
B_A = k_off_A



# ---------------------------------------------------------
# Debye screening
# ---------------------------------------------------------

def debye_kappa():

    charge_sum = (
        (e)**2 * c_Na +
        (e)**2 * c_Cl
    )
    # kappa_D only involved monovalent salt !!
    charge_sum *= (NA * 1000.0) 

    return np.sqrt(charge_sum/(epsilon*kB*T))


kappa_D = debye_kappa()


# Mg contribution to screening
kappa_M2 = (2*e)**2 * c_M * NA *1000.0/ (epsilon*kB*T)


# ---------------------------------------------------------
# Gamma_E(k)
# ---------------------------------------------------------

Gamma_E = D_M * kappa_M2 * (k**2/(k**2 + kappa_D**2))


# ---------------------------------------------------------
# Total transport rate
# ---------------------------------------------------------

Gamma_M = D_M*k**2 + Gamma_E


# ---------------------------------------------------------
# PNP-reaction matrix
# ---------------------------------------------------------

J = np.array([
    [-(Gamma_M + a_D + a_A), B_D, B_A],
    [a_D, -B_D, 0],
    [a_A, 0, -B_A]
])


# ---------------------------------------------------------
# Exact eigenvalues
# ---------------------------------------------------------

eigvals_unsorted = np.linalg.eigvals(J)
eigvals = np.sort(eigvals_unsorted)[::-1]

freq_exact = np.abs(eigvals)/(2*np.pi)


# ---------------------------------------------------------
# Approximate eigenvalues
# ---------------------------------------------------------

ratio = (B_D*B_A)/(a_D*B_A + a_A*B_D + B_D*B_A)

lambda_slow = -Gamma_M * ratio
lambda_D = -(a_D + B_D)
lambda_A = -(a_A + B_A)

eigvals_approx=np.array([lambda_slow, lambda_D,lambda_A])

freq_approx = np.abs(np.array([
    lambda_slow,
    lambda_D,
    lambda_A
]))/(2*np.pi)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
print("Derived Parameters:")
print("S_free  =", S_free ," (M)")
print("theta   =", theta)
print("Debye length  =",1e9/kappa_D," (nm)")
print("Debye kappa   =", kappa_D, " (1/m)")
print("Gamma_E =", Gamma_E, " (1/s)")
print("Gamma_M =", Gamma_M, " (1/s)")
print("a_D     =", a_D, " (1/s)")
print("a_A     =", a_A, " (1/s)")
print("B_D     =", B_D, " (1/s)")
print("B_A     =", B_A, " (1/s)") 
print("ratio   =", ratio)
print("DM      =", D_M, " (m^2/s)")
print("k       =", k, " (1/m)")

print("\nApproximate eigenvalues (s^-1)")
print(eigvals_approx)

print("\nExact eigenvalues (s^-1)")
print(eigvals)

print("\nExact frequencies (Hz)")
print(freq_exact)

print("\nApproximate frequencies (Hz)")
print(freq_approx)

print("\nSensitivity scan\n")

for D_M in [1e-12,3e-12,1e-11,3e-11,1e-10]:

    for k in [5e6,1e7,2e7,1e8]:

        Gamma_E = D_M * kappa_M2 * (k**2/(k**2 + kappa_D**2))
        Gamma_M = D_M*k**2 + Gamma_E

        J = np.array([
            [-(Gamma_M + a_D + a_A), B_D, B_A],
            [a_D, -B_D, 0],
            [a_A, 0, -B_A]
        ])

        eigvals = np.linalg.eigvals(J)
        slow = np.min(np.abs(eigvals))
        freq = slow/(2*np.pi)
        Rloop = (2*np.pi/k)/1e-9

        print(f"D={D_M:.1e}, k={k:.1e}, R={Rloop:.1e}  ->  f_slow={freq:8.2f} Hz")
