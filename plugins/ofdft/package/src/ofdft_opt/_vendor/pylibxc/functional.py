import numpy as np

RHO_MIN = 1e-12
H = 1e-30  # complex step

# ---------------- energy densities f(rho, sigma) = rho * eps_xc ----------------
def _ex_lda(rho):
    return -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (4.0 / 3.0)

def _rs(rho):
    return (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

def _ec_pz(rho):  # Perdew-Zunger 1981, unpolarized (as lda_c_pz in LibXC)
    rs = _rs(rho)
    g, b1, b2 = -0.1423, 1.0529, 0.3334
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    hi = g / (1.0 + b1 * np.sqrt(rs) + b2 * rs)
    lo = A * np.log(rs) + B + C * rs * np.log(rs) + D * rs
    return rho * np.where(rs.real >= 1.0, hi, lo)

def _eps_pw92(rs):  # Perdew-Wang 92 (PW_MOD parameters, as used by PBE in LibXC)
    A, a1, b1, b2, b3, b4 = 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    den = 2.0 * A * (b1 * np.sqrt(rs) + b2 * rs + b3 * rs ** 1.5 + b4 * rs ** 2)
    return -2.0 * A * (1.0 + a1 * rs) * np.log(1.0 + 1.0 / den)

def _ex_pbe(rho, sigma):
    kappa, mu = 0.804, 0.2195149727645171
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    s2 = sigma / (4.0 * kf ** 2 * rho ** 2)
    fx = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa)
    return _ex_lda(rho) * fx

def _ec_pbe(rho, sigma):
    beta = 0.06672455060314922
    gamma = (1.0 - np.log(2.0)) / np.pi ** 2
    ec = _eps_pw92(_rs(rho))
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    ks = np.sqrt(4.0 * kf / np.pi)
    t2 = sigma / (4.0 * ks ** 2 * rho ** 2)
    A = beta / gamma / (np.exp(-ec / gamma) - 1.0)
    at2 = A * t2
    Hc = gamma * np.log(1.0 + beta / gamma * t2 * (1.0 + at2) / (1.0 + at2 + at2 ** 2))
    return rho * (ec + Hc)

_FUNCS = {
    "lda_x":     (lambda r, s: _ex_lda(r), False),
    "lda_c_pz":  (lambda r, s: _ec_pz(r), False),
    "gga_x_pbe": (_ex_pbe, True),
    "gga_c_pbe": (_ec_pbe, True),
}


class LibXCFunctional:
    _needs_laplacian = False

    def __init__(self, name, polarization="unpolarized"):
        if name not in _FUNCS:
            raise KeyError(f"pylibxc stand-in: '{name}' not implemented ({list(_FUNCS)})")
        if polarization != "unpolarized":
            raise NotImplementedError("pylibxc stand-in: only unpolarized")
        self.name = name
        self._f, self.is_gga = _FUNCS[name]

    def compute(self, inp, do_exc=True, do_vxc=True, **kwargs):
        rho = np.asarray(inp["rho"], dtype=float).ravel()
        sig = np.asarray(inp.get("sigma", np.zeros_like(rho)), dtype=float).ravel()
        m = rho > RHO_MIN
        r, s = rho[m], np.maximum(sig[m], 0.0)
        out = {}
        if do_exc:
            zk = np.zeros_like(rho)
            zk[m] = self._f(r, s) / r
            out["zk"] = zk[:, None]
        if do_vxc:
            vr = np.zeros_like(rho)
            vr[m] = self._f(r + 1j * H, s).imag / H
            out["vrho"] = vr[:, None]
            if self.is_gga:
                vs = np.zeros_like(rho)
                vs[m] = self._f(r, s + 1j * H).imag / H
                out["vsigma"] = vs[:, None]
        return out
