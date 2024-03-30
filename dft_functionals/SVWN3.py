import torch

from utils import catch_nan


# VWN
fpp_vwn = 4 / (
    9 * (2 ** (1 / 3) - 1)
)


def Q_vwn(b, c):
    Q = 4 * c - b**2
    x = torch.sqrt(Q)
    catch_nan(b=b, c=c, x=x)
    return x


def f1_vwn(b, c):
    x = 2 * b / Q_vwn(b, c)
    catch_nan(x=x)
    return x


def f2_vwn(b, c, x0):
    x = b * x0 / (x0**2 + b * x0 + c)
    catch_nan(x=x, x0=x0)
    return x


def f3_vwn(b, c, x0):
    x = 2 * (2 * x0 + b) / Q_vwn(b, c)
    catch_nan(x=x)
    return x


def fx_vwn(b, c, rs):
    x = rs + b * torch.sqrt(rs) + c
    catch_nan(x=x)
    return x


def f_aux(A, b, c, x0, rs):
    log_arg = rs / fx_vwn(b, c, rs)
    log = torch.log(log_arg)
    f_vwn = f1_vwn(b, c) - f2_vwn(b, c, x0) * f3_vwn(b, c, x0)
    arc_arg = Q_vwn(b, c) / (2 * torch.sqrt(rs) + b)
    arc = torch.arctan(arc_arg)
    part1 = f_vwn * arc
    log2_arg = (torch.sqrt(rs) - x0) ** 2 / fx_vwn(b, c, rs)
    log2 = torch.log(log2_arg)
    last_part = f2_vwn(b, c, x0) * log2

    x = A * (log + part1 - last_part)

    catch_nan(
        A=A,
        log=log,
        log_arg=log_arg,
        f_vwn=f_vwn,
        arc_arg=arc_arg,
        arc=arc,
        part1=part1,
        log2=log2,
        last_part=last_part,
    )
    return x


def DMC(rs, z, c_arr):
    x = f_aux(
        c_arr[:, 0:2][:, 1],
        c_arr[:, 2:4][:, 1],
        c_arr[:, 4:6][:, 1],
        c_arr[:, 6:8][:, 1],
        rs,
    ) - f_aux(
        c_arr[:, 0:2][:, 0],
        c_arr[:, 2:4][:, 0],
        c_arr[:, 4:6][:, 0],
        c_arr[:, 6:8][:, 0],
        rs,
    )
    catch_nan(x=x)
    return x


def DRPA(rs, z, c_arr):
    x = f_aux(
        c_arr[:, 8:11][:, 1],
        c_arr[:, 11:14][:, 1],
        c_arr[:, 14:17][:, 1],
        c_arr[:, 17:20][:, 1],
        rs,
    ) - f_aux(
        c_arr[:, 8:11][:, 0],
        c_arr[:, 11:14][:, 0],
        c_arr[:, 14:17][:, 0],
        c_arr[:, 17:20][:, 0],
        rs,
    )
    catch_nan(x=x)
    return x


# VWN3


def f_zeta(z):  # - power threshold
    x = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    catch_nan(x=x)
    return x


def f_vwn(rs, z, c_arr):
    aux1 = f_aux(
        c_arr[:, 0:2][:, 0],
        c_arr[:, 2:4][:, 0],
        c_arr[:, 4:6][:, 0],
        c_arr[:, 6:8][:, 0],
        rs,
    )
    dmc = DMC(rs, z, c_arr)
    drpa = DRPA(rs, z, c_arr)
    aux2 = f_aux(
        c_arr[:, 8:11][:, 2],
        c_arr[:, 11:14][:, 2],
        c_arr[:, 14:17][:, 2],
        c_arr[:, 17:20][:, 2],
        rs,
    )
    zeta = f_zeta(z)
    x = (
        aux1
        + torch.nan_to_num(dmc / drpa) * aux2 * zeta * (1 - z**4) / fpp_vwn
        + dmc * zeta * z**4
    )
    catch_nan(aux1=aux1, dmc=dmc, drpa=drpa, aux2=aux2, zeta=zeta, x=x, z=z)
    return x


def rs_z_calc(rho):
    eps = 1e-29
    rs = (3 / ((rho[:, 0] + rho[:, 1] + eps) * (4 * torch.pi))) ** (1 / 3)
    z = (rho[:, 0] - rho[:, 1]) / (rho[:, 0] + rho[:, 1] + eps)
    catch_nan(rs=rs, z=z)
    return rs, z


# SLATER

LDA_X_FACTOR = -3 / 8 * (3 / torch.pi) ** (1 / 3) * 4 ** (2 / 3)  # param
RS_FACTOR = (3 / (4 * torch.pi)) ** (1 / 3)
DIMENSIONS = 3


def f_lda_x(rs, z, c_arr):  # - screen_dens threshold
    x = c_arr[:, 20] * lda_x_spin(rs, z) + c_arr[:, 20] * lda_x_spin(rs, -z)
    catch_nan(x=x)
    return x


def f_xalpha_x(rs, z, constant):
    x = constant[:, 0] * lda_x_spin(rs, z) + constant[:, 0] * lda_x_spin(rs, -z)
    catch_nan(x=x)
    return x


def lda_x_spin(rs, z):
    x = (
        LDA_X_FACTOR
        * (z + 1) ** (1 + 1 / DIMENSIONS)
        * 2 ** (-1 - 1 / DIMENSIONS)
        * (RS_FACTOR / rs)
    )
    catch_nan(x=x)
    return x


def f_svwn3(rho, c_arr):
    """
    rho.shape = (x, 2)
    c_arr.shape = (x, 21)
    """
    catch_nan(rho=rho, c_arr=c_arr)
    rs, z = rs_z_calc(rho)
    return f_lda_x(rs, z, c_arr) + f_vwn(rs, z, c_arr)


def F_XALPHA(rho, constant):
    catch_nan(rho=rho, constant=constant)
    rs, z = rs_z_calc(rho)
    res_energy = f_xalpha_x(rs, z, constant)
    return res_energy


if __name__ == "__main__":
    constants_10 = torch.tile(
        torch.Tensor(
            [
                0.0310907,
                0.01554535,
                3.72744,
                7.06042,
                12.9352,
                18.0578,
                -0.10498,
                -0.32500,
                0.0310907,
                0.01554535,
                -1 / (6 * 3.1415926**2),
                13.0720,
                20.1231,
                1.06835,
                42.7198,
                101.578,
                11.4813,
                -0.409286,
                -0.743294,
                -0.228344,
                1,
            ]
        ),
        (10, 1),
    )

    rho = torch.tile(torch.Tensor([0.3000, 0.0000]), (10, 1))

    print(f_svwn3(rho, constants_10))
