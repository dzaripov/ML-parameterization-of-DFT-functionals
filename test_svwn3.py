from SVWN3 import f_svwn3
import pylibxc as xc
import numpy as np

def calc_fxc(grid): #LDA_C_VWN_3 and "LDA_X"
    func_slater = xc.LibXCFunctional("LDA_X", "polarized")
    func_vwn3 = xc.LibXCFunctional("LDA_C_VWN_3", "polarized")

    inp = {}
    inp["rho"] = grid

    retc_slater = func_slater.compute(inp)
    retc_vwn3 = func_vwn3.compute(inp)

    return retc_slater["zk"], retc_vwn3["zk"]

if __name__ == "__main__":
    for i in range(100):
        data = np.random.random((2))
        exc_libxc_slater, exc_libxc_vwn3 = calc_fxc(data)
        exc_libxc = exc_libxc_slater + exc_libxc_vwn3
        exc_my = f_svwn3(data)
        print(exc_libxc-exc_my)
