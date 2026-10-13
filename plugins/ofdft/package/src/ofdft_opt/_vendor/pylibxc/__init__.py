"""Minimal NumPy stand-in for the `pylibxc` bindings (not the real LibXC).

Provides only what DFTpy uses -- LibXCFunctional(name, "unpolarized").compute()
returning zk, vrho, vsigma -- for: lda_x, lda_c_pz, gga_x_pbe, gga_c_pbe.
Derivatives are taken by complex-step differentiation (exact to round-off).
"""
