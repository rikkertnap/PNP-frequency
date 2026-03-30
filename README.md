# PNP-frequency

## Description

Using the PNP-R equations this set of Python programs 
calculates the (i) relaxation time of an ion solution containing free monovalents ions and
free Mg divalent ion that reversible bind with DNA-phosphates and ATP, and (ii) the change in concentration of 
free, ATP and DNA bound Mg concentration, (iii) Mg release rates,  (iv) and response functions: such as the conducatance.

### PNP-eigenvalue.py :
Computes eigenvalues of Jacobian matrix.

### PNP-spectrum.py :
Computes and plots slow frequency versus loop size for difference D_M.

### PNP-Mgrelease.py :
Computes and plots Mg-release rates from DNA and ATP versus driven frequency.

### PNP-deltaconcentration.py :
Computes and plots the change in free, ATP and DNA bound Mg concentration  versus driven frequency.
Computes and plots the phase lag between Mg binding and unbinding for ATP and DNA, as well

### PNP-conductivity.py :
Computes conductivity and admittance  versus driven frequency.

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

version 03-30-2026

## Authors

* **Rikkert J Nap**
