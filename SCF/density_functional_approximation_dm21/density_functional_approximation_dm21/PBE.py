import torch
#from utils import catch_nan

# PBE C

# c_arr[:, 0] equals mbeta 
# c_arr[:, 1] equals mgamma 
# c_arr[:, 2] equals fz20  # maybe negative?
# c_arr[:, 3:6] equal params_a_beta1
# c_arr[:, 6:9] equal params_a_beta2
# c_arr[:, 9:12] equal params_a_beta3
# c_arr[:, 12:15] equal params_a_beta4 
# c_arr[:, 15:18] equal params_a_a 
# c_arr[:, 18:21] equal params_a_alpha1


#mbeta  = 0.06672455060314922
#mgamma = (1 - torch.log(torch.Tensor([2])))/(torch.pi**2) = 0.0310906916856765747070312
#fz20   = 1.709920934161365617563962776245
#params_a_beta1  = [7.5957, 14.1189, 10.357]
#params_a_beta2  = [3.5876, 6.1977, 3.6231]
#params_a_beta3  = [1.6382, 3.3662,  0.88026]
#params_a_beta4  = [0.49294, 0.62517, 0.49671]
#params_a_a      = [0.031091, 0.015545, 0.016887]
#params_a_alpha1 = [0.21370,  0.20548,  0.11125]


### params_a_pp     = [1,  1,  1]
### c_arr[:, 15:18] equal params_a_pp 


def z_thr(zeta):
    return zeta


def rs_z_calc(rho):
    rs = (3/((rho[:,0] + rho[:,1]) * (4 * torch.pi))) ** (1/3)
    z = z_thr((rho[:,0] - rho[:,1]) / (rho[:,0] + rho[:,1]))
#    catch_nan(rs=rs, z=z)
    return rs, z


def xs_xt_calc(rho, sigmas):     # sigma 1 is alpha beta contracted gradient
    eps = 1e-29
    DIMENSIONS = 3
    xs0 = torch.sqrt(sigmas[:,0])/rho[:,0]**(1 + 1/DIMENSIONS)
    # xs1 = torch.sqrt(sigmas[:,2])/(rho[:,1]+eps)**(1 + 1/DIMENSIONS)
    xs1 = torch.where((sigmas[:,2] < eps) & (rho[:,1] < eps), # last sigma and last rho equal 0
                      torch.sqrt(sigmas[:,0])/rho[:,0]**(1 + 1/DIMENSIONS), 
                      torch.sqrt(sigmas[:,2])/rho[:,1]**(1 + 1/DIMENSIONS))
    xt  = torch.sqrt(sigmas[:,0] + 2*sigmas[:,1] + sigmas[:,2])/(rho[:,0] + rho[:,1])**(1 + 1/DIMENSIONS)

#    catch_nan(rho=rho, sigmas=sigmas, xs0=xs0, xs1=xs1, xt=xt)
    return xs0, xs1, xt


def f_zeta(z): # - power threshold
    res_f_zeta = ((1 + z)**(4/3) + (1 - z)**(4/3) - 2)/(2**(4/3) - 2)
#    catch_nan(res_f_zeta=res_f_zeta)
    return res_f_zeta


def mphi(z):
    res_mphi = ((1 + z)**(2/3) + (1 - z)**(2/3))/2
#    catch_nan(res_mphi=res_mphi)
    return res_mphi
                                    
                                    
def tt(rs, z, xt):
    res_tt = xt/(4*2**(1/3)*mphi(z)*torch.sqrt(rs))
#    catch_nan(res_tt=res_tt)
    return res_tt


def g_aux(k, rs, c_arr):
    res_g_aux = c_arr[:, 3:6][:, k]*torch.sqrt(rs) + c_arr[:, 6:9][:, k]*rs + c_arr[:, 9:12][:, k]*rs**1.5 + c_arr[:, 12:15][:, k]*rs**2
#    catch_nan(res_g_aux=res_g_aux, rs=rs, c_arr=c_arr)
    return res_g_aux


def g(k, rs, c_arr):
    eps = 1e-6
    g_aux_ = g_aux(k, rs, c_arr)
    log = torch.log1p(1/(2*c_arr[:, 15:18][:, k]*g_aux_)) + eps
    res_g = -2*c_arr[:, 15:18][:, k]*(1 + c_arr[:, 18:21][:, k]*rs) * log
#    catch_nan(res_g=res_g, log=log, g_aux_=g_aux_)
    # save_tensors(res_g=res_g, log=log, g_aux_=g_aux_)
    return res_g


def f_pw(rs, z, c_arr):
    res_f_pw = g(0, rs, c_arr) + z**4*f_zeta(z)*(g(1, rs, c_arr) - g(0, rs, c_arr) + g(2, rs, c_arr)/c_arr[:, 2]) - f_zeta(z)*g(2, rs, c_arr)/c_arr[:, 2]
    
    # save_tensors(g0=g(0, rs, c_arr), z_tensor=z**4*f_zeta(z), g1=g(1, rs, c_arr), g2=g(2, rs, c_arr))
#    catch_nan(res_f_pw=res_f_pw)
    return res_f_pw
    

def A(rs, z, t, c_arr, device):
    # eps = 1e-10
    f_pw_ = f_pw(rs, z, c_arr)
    mphi_ = c_arr[:, 1]*mphi(z)**3
    const_87 = torch.Tensor([87]).to(device) # exp(87) = 10**38 - near infinity
    expm1 = torch.expm1(torch.where(-f_pw_/mphi_ < const_87, -f_pw_/mphi_, const_87))
    res_A = (c_arr[:, 0]/(c_arr[:, 1]*expm1))
#    catch_nan(res_A=res_A, f_pw_=f_pw_, mphi_=mphi_, expm1=expm1, rs=rs, z=z, c_arr=c_arr)
    return res_A


def f1(rs, z, t, A_, c_arr):
    BB = 1 # flexibility of A(rs, z, t)*t**4 term \ constant \ 1 or 0
    t2 = t**2
    res_f1 = t2 + BB*A_*t2**2
#    catch_nan(res_f1=res_f1)
    return res_f1


def f2(rs, z, t, c_arr, device):
    eps = 1e-10
    A_ = A(rs, z, t, c_arr, device)
    f1_ = f1(rs, z, t, A_, c_arr)
    res_f2 = c_arr[:, 0]*f1_/(c_arr[:, 1]*(A_*f1_+1) + eps)
#    catch_nan(res_f2=res_f2, f1_=f1_, A_=A_)
    return res_f2


def fH(rs, z, t, c_arr, device):
    eps = 10e-6
    f2_ = f2(rs, z, t, c_arr, device)
    log = torch.where(f2_ <= -1, torch.log1p(f2_ + eps), torch.log1p(f2_)) # weird infinity
    res_fH = c_arr[:, 1]*mphi(z)**3*log
#    catch_nan(res_fH=res_fH, log=log, f2_=f2_)
    return res_fH


def PBE_C(rs, z, xt, c_arr, device):
    res_PBE_C = f_pw(rs, z, c_arr) + fH(rs, z, tt(rs, z, xt), c_arr, device)
#    catch_nan(res_PBE_C=res_PBE_C)
    return res_PBE_C

                            
                                         
#LDA
# c_arr[:, 21] equals LDA_X_FACTOR 

#LDA_X_FACTOR = -3/8*(3/torch.pi)**(1/3)*4**(2/3) # param


def lda_x_spin(rs, z, c_arr):
    RS_FACTOR = (3/(4*torch.pi))**(1/3)
    DIMENSIONS = 3
    rs_f_rs = (RS_FACTOR/rs)
    res_lda_x_spin = c_arr[:, 21]*(z+1)**(1 + 1/DIMENSIONS)*2**(-1-1/DIMENSIONS)*rs_f_rs
#    catch_nan(res_lda_x_spin=res_lda_x_spin, const=c_arr[:, 21], rs_f_rs=rs_f_rs)
    return res_lda_x_spin


                                         
# PBE X

# c_arr[:, 22] equals params_a_kappa 
# c_arr[:, 23] equals params_a_mu 


# params_a_kappa = 0.8040
# params_a_mu    = 0.2195149727645171




def pbe_f0(s, c_arr):
    res_pbe_f0 = 1 + c_arr[:, 22]*(1 - c_arr[:, 22]/(c_arr[:, 22] + c_arr[:, 23]*s**2))
#    catch_nan(res_pbe_f0=res_pbe_f0)
    return res_pbe_f0


def pbe_f(x, c_arr):
    X2S = 1/(2*(6*torch.pi**2)**(1/3))
    res_pbe_f = pbe_f0(X2S*x, c_arr)
#    catch_nan(res_pbe_f=res_pbe_f)
    return res_pbe_f


def gga_exchange(func, rs, z, xs0, xs1, c_arr): # -screen_dens -z_thr
    res_gga_exchange = lda_x_spin(rs, z, c_arr)*func(xs0, c_arr) + lda_x_spin(rs, -z, c_arr)*func(xs1, c_arr)
#    catch_nan(res_gga_exchange=res_gga_exchange)
    return res_gga_exchange 

                                         
def PBE_X(rs, z, xt, xs0, xs1, c_arr):
    res_PBE_X = gga_exchange(pbe_f, rs, z, xs0, xs1, c_arr)
#    catch_nan(res_PBE_X=res_PBE_X)
    return res_PBE_X


# @torch.compile
def F_PBE(rho, sigmas, c_arr, device):
#    catch_nan(rho=rho, sigmas=sigmas, c_arr=c_arr)
    rs, z = rs_z_calc(rho)
    xs0, xs1, xt = xs_xt_calc(rho, sigmas)
    res_energy = PBE_X(rs, z, xt, xs0, xs1, c_arr) + PBE_C(rs, z, xt, c_arr, device)
#    catch_nan(res_energy=res_energy)
    return res_energy


def pw_test(rho, c_arr):
    rs, z = rs_z_calc(rho)
    pw_energy = f_pw(rs, z, c_arr)
#    catch_nan(pw_energy=pw_energy)
    return pw_energy