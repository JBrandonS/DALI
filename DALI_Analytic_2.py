# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 03 2022 04:34 PM

@author: Owner
"""

from numpy import log, matrix, loadtxt, exp, arange, savetxt, array, random, inf, isfinite, pi, log10, sin, sinh, sqrt, cos, tanh, cosh, identity, outer
from scipy.integrate import quad, dblquad
from numpy.linalg import inv
from math import factorial
from sympy import symbols, diff, lambdify, var, Array
from sympy.functions import exp

N = 4
if N == 1:
    #H(z) alone:
    BFP_H0 = 68.1461
    BFP_Om = 0.319671
    lab = 'H(z)'

if N == 2:
    #H(z)+6BAO:
    BFP_H0 = 68.8653
    BFP_Om = 0.313987
    lab = 'H(z) + 6BAO'

if N == 3:
    #H(z)+6BAO+QSO
    BFP_H0 = 68.8261
    BFP_Om = 0.313344
    lab = 'H(z) + 6BAO + QSO'
    
if N == 4:
    #QSO alone:
    BFP_H0 = 68.7007
    BFP_Om = 0.315243
    lab = 'QSO'

if N == 5:
    #6BAO+QSO:
    BFP_H0 = 68.9356
    BFP_Om = 0.314931
    lab = '6BAO + QSO'

if N == 6:
    #BAO alone
    BFP_H0 = 71.3695
    BFP_Om = 0.364813
    lab = '6BAO'

###
#H(z) section
###
c = 299792458/1000 #Speed of light in km/s
Obh2 = 0.02225
On = 0.0014

def E(z, O, L): #Expansion parameter
    return (O*((1 + z)**3) + (1 - O - L)*((1 + z)**2) + L)**(1/2)

def H(H0, z, O, L): #Hubble parameter
    return H0*E(z, O, L)

z_obs, Hz_obs, sigHobs = loadtxt('H(z)data.dat',unpack = True)
##From Table 1 of 1607.03537v2, refs. 4,6,7,10 excluded.
    
def H_1(H0, z, O, L):
    A = ((1 + z)**3) - 1
    B = 2*E(z, O, L)
    return H0*(A/B)
    
def H_11(H0, z, O, L):
    A = (((1 + z)**3) - 1)**2
    B = 4*(E(z, O, L)**3)
    return -H0*(A/B)
    
def H_10(H0, z, O, L):
    A = ((1 + z)**3) - 1
    B = 2*E(z, O, L)
    return A/B
    
def H_01(H0, z, O, L):
    return H_10(H0, z, O, L)
    
def H_00(H0, z, O, L):
    return 0
    
Id = identity(len(sigHobs))
M = outer((1/sigHobs), (1/sigHobs))*Id

H_1_list = []
H_0_list = []

H_11_list = []
H_10_list = []
H_01_list = []
H_00_list = []

for z in z_obs:
    H_1_list.append(H_1(BFP_H0, z, BFP_Om, 1 - BFP_Om))
    H_0_list.append(E(z, BFP_Om, 1-BFP_Om))

    H_11_list.append(H_11(BFP_H0, z, BFP_Om, 1 - BFP_Om))
    H_10_list.append(H_10(BFP_H0, z, BFP_Om, 1 - BFP_Om))
    H_01_list.append(H_01(BFP_H0, z, BFP_Om, 1 - BFP_Om))
    H_00_list.append(H_00(BFP_H0, z, BFP_Om, 1 - BFP_Om))
    
H_1_arr = array(H_1_list)
H_0_arr = array(H_0_list)

H_11_arr = array(H_11_list)
H_10_arr = array(H_10_list)
H_01_arr = array(H_01_list)
H_00_arr = array(H_00_list)

prod_1 = (M.dot(H_1_arr)).T
prod_0 = (M.dot(H_0_arr)).T
                     
prod_11 = (M.dot(H_11_arr)).T
prod_01 = (M.dot(H_01_arr)).T
prod_10 = (M.dot(H_10_arr)).T
prod_00 = (M.dot(H_00_arr)).T

M_00_Hz = H_0_arr.dot(prod_0)
M_01_Hz = H_0_arr.dot(prod_1)
M_10_Hz = H_1_arr.dot(prod_0)
M_11_Hz = H_1_arr.dot(prod_1)

M_000_Hz = H_00_arr.dot(prod_0)
M_001_Hz = H_00_arr.dot(prod_1)
M_010_Hz = H_01_arr.dot(prod_0)
M_100_Hz = H_10_arr.dot(prod_0)
M_101_Hz = H_10_arr.dot(prod_1)
M_110_Hz = H_11_arr.dot(prod_0)
M_011_Hz = H_01_arr.dot(prod_1)
M_111_Hz = H_11_arr.dot(prod_1)

M_0000_Hz = H_00_arr.dot(prod_00)
M_0001_Hz = H_00_arr.dot(prod_01)
M_0010_Hz = H_00_arr.dot(prod_10)
M_0011_Hz = H_00_arr.dot(prod_11)
M_0100_Hz = H_01_arr.dot(prod_00)
M_0101_Hz = H_01_arr.dot(prod_01)
M_0110_Hz = H_01_arr.dot(prod_10)
M_0111_Hz = H_01_arr.dot(prod_11)
M_1000_Hz = H_10_arr.dot(prod_00)
M_1001_Hz = H_10_arr.dot(prod_01)
M_1010_Hz = H_10_arr.dot(prod_10)
M_1011_Hz = H_10_arr.dot(prod_11)
M_1100_Hz = H_11_arr.dot(prod_00)
M_1101_Hz = H_11_arr.dot(prod_01)
M_1110_Hz = H_11_arr.dot(prod_10)
M_1111_Hz = H_11_arr.dot(prod_11)

###
#QSO section
###
z_obsQ, th_obs, sig_th_obs = loadtxt('QSO_120.txt', unpack=True)

c = 299792458/1000 #Speed of light in km/s
#Th = 2.7255/2.7 #T_CMB/2.7, from Eisenstein and Hu 1998
                #and Fixsen 0911.1955
lm = 11.03 #Standard rod in units of pc
    
def D_M(H0, z, O, L): #Transverse comoving distance (see astro-ph/9905116v4)
    dH = c/H0
    I, error = quad(lambda m: 1/(E(m, O, L)), 0, z)
    if 1-O-L < 0:            
        y = (1/((-(1-O-L))**(1/2)))*(sin(((-(1-O-L))**(1/2))*I))
    if 1-O-L > 0:            
        y = (1/(((1-O-L))**(1/2)))*(sinh((((1-O-L))**(1/2))*I))
    if 1-O-L == 0:
        y = I
    return y*dH
    
def th(H0, z, O, L): #Eq. (2) of 1708.08635v1
    #D_A = D_M(1 + z) (see astro-ph/9905116v4) so l_m/D_A = l_m(1 + z)/D_M.
    #D_M has units of Mpc, so the 10^6 factor converts it to pc.
    #1 rad = 2.06265*(10^8) milliarcsec.
    return ((180/pi)*(3600*1000))*(1 + z)*(lm/((10**6)*D_M(H0, z, O, L)))
    
def dDM_dOm(z, H0, O):
    I, error = quad(lambda m: (((1 + m)**3) - 1)/((E(m, O, 1-O))**3), 0, z)
    return (-c/(2*H0))*I
    
def dDM_dH0(z, H0, O):
    I, error = quad(lambda m: 1/(E(m, O, 1-O)), 0, z)
    return (-c/(H0**2))*I
    
def DM_00(z, H0, O):
    I, error = quad(lambda m: 1/(E(m, O, 1-O)), 0, z)
    return (2*c/(H0**3))*I
    
def DM_10(z, H0, O):
    I, error = quad(lambda m: (((1 + m)**3) - 1)/((E(m, O, 1-O))**3), 0, z)
    return (c/(2*(H0**2)))*I
    
def DM_01(z, H0, O):
    return DM_10(z, H0, O)

def DM_11(z, H0, O):
    I, error = quad(lambda m: ((((1 + m)**3) - 1)**2)/((E(m, O, 1-O))**5), 0, z)
    return (3*c/(4*H0))*I
    
def th_0(z, H0, O):
    return -(th(H0, z, O, 1-O)/D_M(H0, z, O, 1-O))*(dDM_dH0(z, H0, O))
    
def th_1(z, H0, O):
    return -(th(H0, z, O, 1-O)/D_M(H0, z, O, 1-O))*(dDM_dOm(z, H0, O))
    
def th_00(z, H0, O):
    #A = (-2/D_M(H0, z, O, 1-O))*th_0(z, H0, O)*dDM_dH0(z, H0, O)
    #B = (-th(H0, z, O, 1-O)/D_M(H0, z, O, 1-O))*DM_00(z, H0, O)
    #return A + B
    A = 2*dDM_dH0(z, H0, O)*dDM_dH0(z, H0, O)
    B = D_M(H0, z, O, 1-O)*DM_00(z, H0, O)
    C = th(H0, z, O, 1-O)/(D_M(H0, z, O, 1-O)**2)
    return C*(A - B)
    
def th_01(z, H0, O):
    #A = (-2/D_M(H0, z, O, 1-O))*th_1(z, H0, O)*dDM_dH0(z, H0, O)
    #B = (-th(H0, z, O, 1-O)/D_M(H0, z, O, 1-O))*DM_01(z, H0, O)
    #return A + B
    A = 2*dDM_dH0(z, H0, O)*dDM_dOm(z, H0, O)
    B = D_M(H0, z, O, 1-O)*DM_01(z, H0, O)
    C = th(H0, z, O, 1-O)/(D_M(H0, z, O, 1-O)**2)
    return C*(A - B)
    
def th_10(z, H0, O):
    return th_01(z, H0, O)
    
def th_11(z, H0, O):
    #A = (-2/D_M(H0, z, O, 1-O))*th_1(z, H0, O)*dDM_dOm(z, H0, O)
    #B = (-th(H0, z, O, 1-O)/D_M(H0, z, O, 1-O))*DM_11(z, H0, O)
    #return A + B
    A = 2*dDM_dOm(z, H0, O)*dDM_dOm(z, H0, O)
    B = D_M(H0, z, O, 1-O)*DM_11(z, H0, O)
    C = th(H0, z, O, 1-O)/(D_M(H0, z, O, 1-O)**2)
    return C*(A - B)
    
Id = identity(len(sig_th_obs))
M_QSO = outer((1/(sig_th_obs + 0.1*th_obs)), (1/(sig_th_obs + 0.1*th_obs)))*Id

th_1_list = []
th_0_list = []

th_11_list = []
th_01_list = []
th_10_list = []
th_00_list = []

for z in z_obsQ:
    th_1_list.append(th_1(z, BFP_H0, BFP_Om))
    th_0_list.append(th_0(z, BFP_H0, BFP_Om))
    
    th_11_list.append(th_11(z, BFP_H0, BFP_Om))
    th_01_list.append(th_01(z, BFP_H0, BFP_Om))
    th_10_list.append(th_10(z, BFP_H0, BFP_Om))
    th_00_list.append(th_00(z, BFP_H0, BFP_Om))

th_1_arr = array(th_1_list)
th_0_arr = array(th_0_list)

th_11_arr = array(th_11_list)
th_01_arr = array(th_01_list)
th_10_arr = array(th_10_list)
th_00_arr = array(th_00_list)
                     
prod_1 = (M_QSO.dot(th_1_arr)).T
prod_0 = (M_QSO.dot(th_0_arr)).T

prod_11 = (M_QSO.dot(th_11_arr)).T
prod_01 = (M_QSO.dot(th_01_arr)).T
prod_10 = (M_QSO.dot(th_10_arr)).T
prod_00 = (M_QSO.dot(th_00_arr)).T

M_00_Q = th_0_arr.dot(prod_0)
M_01_Q = th_0_arr.dot(prod_1)
M_10_Q = th_1_arr.dot(prod_0)
M_11_Q = th_1_arr.dot(prod_1)

M_000_Q = th_00_arr.dot(prod_0)
M_001_Q = th_00_arr.dot(prod_1)
M_010_Q = th_01_arr.dot(prod_0)
M_100_Q = th_10_arr.dot(prod_0)
M_101_Q = th_10_arr.dot(prod_1)
M_110_Q = th_11_arr.dot(prod_0)
M_011_Q = th_01_arr.dot(prod_1)
M_111_Q = th_11_arr.dot(prod_1)

M_0000_Q = th_00_arr.dot(prod_00)
M_0001_Q = th_00_arr.dot(prod_01)
M_0010_Q = th_00_arr.dot(prod_10)
M_0011_Q = th_00_arr.dot(prod_11)
M_0100_Q = th_01_arr.dot(prod_00)
M_0101_Q = th_01_arr.dot(prod_01)
M_0110_Q = th_01_arr.dot(prod_10)
M_0111_Q = th_01_arr.dot(prod_11)
M_1000_Q = th_10_arr.dot(prod_00)
M_1001_Q = th_10_arr.dot(prod_01)
M_1010_Q = th_10_arr.dot(prod_10)
M_1011_Q = th_10_arr.dot(prod_11)
M_1100_Q = th_11_arr.dot(prod_00)
M_1101_Q = th_11_arr.dot(prod_01)
M_1110_Q = th_11_arr.dot(prod_10)
M_1111_Q = th_11_arr.dot(prod_11)

if N == 1:
    #G_list = [M_000_Hz, M_001_Hz, M_010_Hz, M_100_Hz, M_101_Hz, M_110_Hz, M_011_Hz, M_111_Hz]
    G_list = [M_000_Hz, M_001_Hz, M_010_Hz, M_011_Hz, M_100_Hz, M_101_Hz, M_110_Hz, M_111_Hz]
    
    H_list = [M_0000_Hz, M_0001_Hz, M_0010_Hz, M_0011_Hz, M_0100_Hz, M_0101_Hz, M_0110_Hz, M_0111_Hz, M_1000_Hz, M_1001_Hz, M_1010_Hz, M_1011_Hz, M_1100_Hz, M_1101_Hz, M_1110_Hz, M_1111_Hz]

    Fisher = [M_00_Hz, M_01_Hz, M_10_Hz, M_11_Hz]

if N == 4:
    G_list = [M_000_Q, M_001_Q, M_010_Q, M_011_Q, M_100_Q, M_101_Q, M_110_Q, M_111_Q]
    
    H_list = [M_0000_Q, M_0001_Q, M_0010_Q, M_0011_Q, M_0100_Q, M_0101_Q, M_0110_Q, M_0111_Q, M_1000_Q, M_1001_Q, M_1010_Q, M_1011_Q, M_1100_Q, M_1101_Q, M_1110_Q, M_1111_Q]
    
    Fisher = [M_00_Q, M_01_Q, M_10_Q, M_11_Q]

H0, O = symbols('H0 O', real=True)

x = [H0, O]

for Trange in range(1, 21, 1):
    sum1 = 0
    count = -1
    for i in range(0, 2, 1):
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                for l in range(0, 2, 1):
                    count += 1
                    sum1 += H_list[count]*x[i]*x[j]*x[k]*x[l]
                    
    ssum1 = sum1 - H_list[0]*x[0]*x[0]*x[0]*x[0] - H_list[15]*x[1]*x[1]*x[1]*x[1]
    
    sssum1 = 0
    for n in range(0, Trange, 1):
        sssum1 += ((-(1/8)*ssum1)**n)/factorial(n)
        
    sum2 = 0
    count = -1
    for i in range(0, 2, 1):
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                count += 1
                sum2 += G_list[count]*x[i]*x[j]*x[k]
    
    ssum2 = 0
    for n in range(0, Trange, 1):
        ssum2 += ((-(1/2)*sum2)**n)/factorial(n)
        
    sum3 = 0
    count = -1
    for i in range(0, 2, 1):
        for j in range(0, 2, 1):
            count += 1
            sum3 += Fisher[count]*x[i]*x[j]
    
    ssum3 = 0
    for n in range(0, Trange, 1):
        ssum3 += ((-(1/2)*sum3)**n)/factorial(n)
        
    Z = exp(-H_list[0]*x[0]*x[0]*x[0]*x[0] - H_list[15]*x[1]*x[1]*x[1]*x[1])*sssum1*ssum2*ssum3
    
    ZZ = lambdify([(H0, O)], Z)
    
    ZZZ = lambda y, x: ZZ((y, x))
    
    I0, error = dblquad(ZZZ, 0.1, 0.7, lambda x: 50, lambda x: 85)
    
    print(Trange, I0, "Z")
    
    H0_var = x[0]*x[0]*Z
    
    H0_VAR = lambdify([(H0, O)], H0_var)
    
    H0_v = lambda y, x: H0_VAR((y, x))
    
    I, error = dblquad(H0_v, 0.1, 0.7, lambda x: 50, lambda x: 85)
    
    print(Trange, I/I0, "<H0*H0>")
    
    H0_O_covar = x[0]*x[1]*Z
    
    H0_O_COVAR = lambdify([(H0, O)], H0_O_covar)
    
    H0_O_cov = lambda y, x: H0_O_COVAR((y, x))
    
    I, error = dblquad(H0_O_cov, 0.1, 0.7, lambda x: 50, lambda x: 85)
    
    print(Trange, I/I0, "<H0*Omega_m0>")
    
    O_H0_covar = x[1]*x[0]*Z
    
    O_H0_COVAR = lambdify([(H0, O)], O_H0_covar)
    
    O_H0_cov = lambda y, x: O_H0_COVAR((y, x))
    
    I, error = dblquad(O_H0_cov, 0.1, 0.7, lambda x: 50, lambda x: 85)
    
    print(Trange, I/I0, "<Omega_m0*H0>")
    
    O_var = x[0]*x[0]*Z
    
    O_VAR = lambdify([(H0, O)], O_var)
    
    O_v = lambda y, x: O_VAR((y, x))
    
    I, error = dblquad(O_v, 0.1, 0.7, lambda x: 50, lambda x: 85)
    
    print(Trange, I/I0, "<Omega_m0*Omega_m0>")

'''
sum1 = 0
count = -1
for i in range(0, 2, 1):
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            for l in range(0, 2, 1):
                count += 1
                sum1 += (-1/8)*H_list[count]*x[i]*x[j]*x[k]*x[l]
#print(sum1)    
sum2 = 0
count = -1
for i in range(0, 2, 1):
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            count += 1
            sum2 += (-1/2)*G_list[count]*x[i]*x[j]*x[k]
    
sum3 = 0
count = -1
for i in range(0, 2, 1):
    for j in range(0, 2, 1):
        count += 1
        sum3 += (-1/2)*Fisher[count]*x[i]*x[j]

Z = exp(sum1 + sum2 + sum3)
print(Z)
ZZ = lambdify([(H0, O)], Z)
print(ZZ((BFP_H0, BFP_Om)))
ZZZ = lambda y, x: ZZ((y, x))

I0, error = dblquad(ZZZ, 0.1, 0.7, lambda x: 50, lambda x: 85)

print("No Taylor:", I0, "Z")

H0_var = x[0]*x[0]*Z

H0_VAR = lambdify([(H0, O)], H0_var)

H0_v = lambda y, x: H0_VAR((y, x))

I, error = dblquad(H0_v, 0.1, 0.7, lambda x: 50, lambda x: 85)

print("No Taylor:", I/I0, "<H0*H0>")

H0_O_covar = x[0]*x[1]*Z

H0_O_COVAR = lambdify([(H0, O)], H0_O_covar)

H0_O_cov = lambda y, x: H0_O_COVAR((y, x))

I, error = dblquad(H0_O_cov, 0.1, 0.7, lambda x: 50, lambda x: 85)

print("No Taylor:", I/I0, "<H0*Omega_m0>")

O_H0_covar = x[1]*x[0]*Z

O_H0_COVAR = lambdify([(H0, O)], O_H0_covar)

O_H0_cov = lambda y, x: O_H0_COVAR((y, x))

I, error = dblquad(O_H0_cov, 0.1, 0.7, lambda x: 50, lambda x: 85)

print("No Taylor:", I/I0, "<Omega_m0*H0>")

O_var = x[0]*x[0]*Z

O_VAR = lambdify([(H0, O)], O_var)

O_v = lambda y, x: O_VAR((y, x))

I, error = dblquad(O_v, 0.1, 0.7, lambda x: 50, lambda x: 85)

print("No Taylor:", I/I0, "<Omega_m0*Omega_m0>")
'''





