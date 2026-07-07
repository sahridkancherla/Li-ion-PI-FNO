"""
Physical constants needed for physics losses. Extracted from Chen2020 parameter set.
"""
import pybamm

F = 96485.3329  # Faraday's constant in C/mol

def get_spm_const(param_vals=None):
    """Extracts the parameter values needed for physics losses.
    Args:
        param_vals: pybamm.ParameterValues. If None, loads Chen2020.
    Returns:
        dict of float constants in SI units - {D_n: negative particle diffusivity,
        D_p: positive particle diffusivity, R_n: negative particle radius,
        R:p positive particle radius, c0_n: initial conc. in negative electrode,
        c0_p: initial conc. in positive electrode}
    """
    if param_vals is None:
        param_vals = pybamm.ParameterValues("Chen2020")
    
    constants = {
    "F": F, # Faraday constant

    "D_n": float(param_vals["Negative particle diffusivity [m2.s-1]"]),
    "D_p": float(param_vals["Positive particle diffusivity [m2.s-1]"]),

    "R_n": float(param_vals["Negative particle radius [m]"]),
    "R_p": float(param_vals["Positive particle radius [m]"]),

    "c0_n": float(param_vals["Initial concentration in negative electrode [mol.m-3]"]),
    "c0_p": float(param_vals["Initial concentration in positive electrode [mol.m-3]"]),

    "c_max_n": float(param_vals["Maximum concentration in negative electrode [mol.m-3]"]),
    "c_max_p": float(param_vals["Maximum concentration in positive electrode [mol.m-3]"]),
}

    return constants

if __name__ == "__main__":
    c = get_spm_const()
    for k, v in c.items():
        print(f"{k}: {v}")
