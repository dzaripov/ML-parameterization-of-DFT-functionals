import numpy as np


A_vwn  = [ 0.0310907, 0.01554535]
b_vwn  = [ 3.72744,   7.06042]
c_vwn  = [12.9352,    18.0578]
x0_vwn = [-0.10498,  -0.32500]

A_rpa  = [ 0.0310907,  0.01554535,  -1/(6*np.pi**2)]
b_rpa  = [13.0720,    20.1231,      1.06835  ]
c_rpa  = [42.7198,   101.578,      11.4813   ]
x0_rpa = [-0.409286,  -0.743294,   -0.228344 ]

params_a_alpha = 1


#VWN
p_a_zeta_threshold = 1e-15

fpp_vwn = 4/(9*(2**(1/3) - 1))

def Q_vwn(b, c):
    return np.sqrt(4*c - b**2)

def f1_vwn(b, c):
    return 2*b/Q_vwn(b, c)

def f2_vwn(b, c, x0):
    return b*x0/(x0**2 + b*x0 + c)

def f3_vwn(b, c, x0):
    return 2*(2*x0 + b)/Q_vwn(b, c)

def fx_vwn(b, c, rs):
    return rs + b*np.sqrt(rs) + c

def opz_pow_n(z, n):
    if 1 + z <= p_a_zeta_threshold:
        return (p_a_zeta_threshold)^n
    else:
        return (1+z)**n


def f_aux(A, b, c, x0, rs):
  return A*(
  + np.log(rs/fx_vwn(b, c, rs))
  + (f1_vwn(b, c) - f2_vwn(b, c, x0)*f3_vwn(b, c, x0))
  * np.arctan(Q_vwn(b, c)/(2*np.sqrt(rs) + b))
  - f2_vwn(b, c, x0)*np.log((np.sqrt(rs) - x0)**2/fx_vwn(b, c, rs)))


def DMC(rs, z):
    return f_aux(A_vwn[1], b_vwn[1], c_vwn[1], x0_vwn[1], rs) \
    - f_aux(A_vwn[0], b_vwn[0], c_vwn[0], x0_vwn[0], rs)

def DRPA(rs, z):
    return f_aux(A_rpa[1], b_rpa[1], c_rpa[1], x0_rpa[1], rs) \
    - f_aux(A_rpa[0], b_rpa[0], c_rpa[0], x0_rpa[0], rs)

#VWN3

def f_zeta(z): # - power threshold
    return ((1 + z)**(4/3) + (1 - z)**(4/3) - 2)/(2**(4/3) - 2)

def f_vwn(rs, z):
  return f_aux(A_vwn[0], b_vwn[0], c_vwn[0], x0_vwn[0], rs) \
  + DMC(rs, z)/DRPA(rs, z)*f_aux(A_rpa[2], b_rpa[2], c_rpa[2], x0_rpa[2], rs)*f_zeta(z)*(1 - z**4)/fpp_vwn \
  + DMC(rs, z)*f_zeta(z)*z**4

def rs_z_calc(rho):
    rs = (3/((rho[0] + rho[1]) * (4 * np.pi))) ** (1/3)
    z = (rho[0] - rho[1]) / (rho[0] + rho[1])
    return rs, z

def f_vwn3(rho):
    rs, z = rs_z_calc(rho)
    return f_vwn(rs, z)


#SLATER

LDA_X_FACTOR = -3/8*(3/np.pi)**(1/3)*4**(2/3)
RS_FACTOR = (3/(4*np.pi))**(1/3)
DIMENSIONS = 3


def f_lda_x(rs, z): # - screen_dens threshold
    return params_a_alpha*lda_x_spin(rs, z) + params_a_alpha*lda_x_spin(rs, -z)

def lda_x_spin(rs, z):
    return LDA_X_FACTOR*(z+1)**(1 + 1/DIMENSIONS)*2**(-1-1/DIMENSIONS)*(RS_FACTOR/rs)

def f_slater(rho):
    rs, z = rs_z_calc(rho)
    return f_lda_x(rs, z)