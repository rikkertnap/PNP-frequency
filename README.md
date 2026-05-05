# PNP-frequency

## Description

Using the PNP-R equations this set of Python programs 
calculates the (i) relaxation time of an ion solution containing free monovalents ions and
free Mg divalent ion that reversible bind with DNA-phosphates and ATP, and (ii) the change in concentration of 
free, ATP and DNA bound Mg concentration, (iii) Mg release rates,  (iv) and response functions: such as the conductance.
 
Program 1-5: Implement the minimal model: only the ions move. <br>
Program 6-10: Refinemnt of minimal model: ATP and DNA can move.  <br>

### 1. PNP-eigenvalue.py :
Computes eigenvalues of Jacobian matrix.

### 2. PNP-spectrum.py :
Computes and plots slow frequency versus loop size for difference D_M.

### 3. PNP-Mgrelease.py :
Computes and plots Mg-release rates from DNA and ATP versus driven frequency.

### 4. PNP-deltaconcentration.py :
Computes and plots the change in free, ATP and DNA bound Mg concentration  versus driven frequency.
Computes and plots the phase lag between Mg binding and unbinding for ATP and DNA, as well

### 5. PNP-conductivity.py :
Computes conductivity and admittance  versus driven frequency.

### 6. PNP-Mgrelease.py :
Enforce charge coupling of rho_fixed
Computes and plots Mg-release rates from DNA and ATP versus driven frequency.

### 7. PNP-ATP-frequency.py :
Computes eigenvalues of Jacobian matrix and relaxtion time/frequencies
Allow movement movement of ATP and DNA not moving.

### 8. PNP-ATP-DNA-frequency.py :
Computes eigenvalues of Jacobian matrix and relaxtion time/frequencies
Allow movement movement of ATP and DNA.

### 9. PNP-ATP-Mg-release.py : 
Computes and plots Mg-release rates from DNA and ATP versus driven frequency.
Allow movement movement of ATP.

### 10. PNP-ATP-deltaconcnetration-release.py :
Computes and plots the change in free, ATP and DNA bound Mg concentration  versus driven frequency.
Allow movement movement of ATP. Note completed yet

### Requirement  

matplotlib, numpy 

### Running

python PNP-eigenvalues.py or python PNP-Mgrelease.py

In all programs, except PNP-eigenvalue.py", which computes only the internal frequencies,
the external driven force is controlled with the variable "extforce". <br>
extforce= "electric" : default electrostatic <br>
extforce= "driven"   : the shape and value of driven force needs to be specfied in function 
external_force.<br>

## Versioning

version 05-05-2026

## Authors

* **Rikkert J Nap**
