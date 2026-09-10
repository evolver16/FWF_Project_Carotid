from CMM_plotter import *
from CMM import *
import jax
import jax.numpy as jnp
from jax import jit

@jit
def F_calc_uniax(lam, e=(1.0, 0.0, 0.0)):
    """F = lam*(e⊗e) + lam**(-1/2)*(I - e⊗e)"""
    e = jnp.array(e)
    e = e / jnp.linalg.norm(e)
    I = jnp.eye(3)
    e_dyad_e = jnp.outer(e, e)
    return lam * e_dyad_e + lam ** (-0.5) * (I - e_dyad_e)

def test_case1():
    T = 101.0
    rho_0 = 1.0
    sigma_0 = 0.0
    k1, k2 = 10.0, 5.0
    direction = (0.0, 0.0, 1.0)
    G = jnp.eye(3)*1.0
    lam_step = 1.05
    ds = 1.0
    n_steps = 50
 
    W = fung_strain_energy(k1, k2, direction)

    fiber = constituent(rho_0, sigma_0, T, G, W, k_minus=0.0, k_plus=0.0)
 
    F0 = F_calc_uniax(1.0, direction)
    F_s = F_calc_uniax(lam_step, direction)
 
    F_history = [F0]
    tau_history = [0.0]
    s = 0.0
    for step in range(n_steps):
        s += ds
        F_history.append(F_s)
        tau_history.append(s)
        fiber.m_history.append(rho_0 / T)
        fiber.append_converged_step(0.0, ds)
 
    F_history_arr = jnp.stack(F_history)
    tau_history_arr = jnp.array(tau_history)
 
    # --- baseline: homeostatic-specific Q/q ---
    psi_homstat = Psi_calc(fiber, s, F_s, F_history_arr, tau_history_arr,
                            Q_homstat_calc, q_homstat_calc)
    rho_homstat = rho_calc(s, fiber, tau_history_arr, Q_homstat_calc, q_homstat_calc)
 
    # --- general K-based Q/q, with k_sigma_minus=0 ---
    psi_general = Psi_calc(fiber, s, F_s, F_history_arr, tau_history_arr,
                            Q_calc, q_calc)
    rho_general = rho_calc(s, fiber, tau_history_arr, Q_calc, q_calc)
 
    C = C_calc(F_s)
    I4 = I4_calc(C, direction)
    W0_val = (k1 / (2 * k2)) * (jnp.exp(k2 * (I4 - 1) ** 2) - 1)
    psi_closed_form = rho_0 * jnp.exp(-s / T) * W0_val
 
    print(f'Psi (homeostatic Q/q):      {float(psi_homstat):.6f}')
    print(f'Psi (general Q/q, k=0):     {float(psi_general):.6f}')
    print(f'Psi (closed form):          {float(psi_closed_form):.6f}')
    print(f'  homstat vs general diff:  {float(abs(psi_homstat - psi_general)):.6e}')
    print(f'  homstat vs closed diff:   {float(abs(psi_homstat - psi_closed_form)):.6e}')
    print()
    print(f'rho (homeostatic Q/q):      {float(rho_homstat):.6f}  (should be ~1.0)')
    print(f'rho (general Q/q, k=0):     {float(rho_general):.6f}  (should be ~1.0)')



def test_case2():
    pass

import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from jax import grad
import jax.numpy as jnp



def total_energy(lam, s, fiber, F_history, tau_history, F_g_s, F_g_history, C10, K, direction):
    F_s = F_calc_uniax(lam, direction)
    C = C_calc(F_s)
    I1 = jnp.trace(C)
    J = jnp.linalg.det(F_s)
    I1_inc = I1 * J**(-2/3)
    psi_e = C10 * (I1_inc - 3) + (K / 2) * (J - 1)**2
    psi_c = Psi_calc(fiber, s, F_s, F_history, tau_history, Q_calc, q_calc, F_g_s, F_g_history)
    return psi_e + psi_c

energy_grad = jit(grad(total_energy, argnums=0))

def get_stress_yy(lam, s, fiber, F_history, tau_history, F_g_s, F_g_history, C10, K, direction):
    return energy_grad(lam, s, fiber, F_history, tau_history, F_g_s, F_g_history, C10, K, direction) * lam

def solve_lam(target_stress, lam_guess, s, fiber, F_hist, tau_hist_full, F_g_s, F_g_hist, C10, K, direction):
    def objective(lam):
        F_s_temp = F_calc_uniax(lam, direction)
        F_hist_temp = jnp.concatenate([F_hist, F_s_temp[None, ...]])
        F_g_hist_temp = jnp.concatenate([F_g_hist, F_g_s[None, ...]])
        return float(get_stress_yy(lam, s, fiber, F_hist_temp, tau_hist_full, F_g_s, F_g_hist_temp, C10, K, direction)) - target_stress
    res = root_scalar(objective, x0=lam_guess, x1=lam_guess + 0.01)
    return res.root

def simulate(case_type, lam_val, stress_val, n_steps, ds, T_val, rho_0, sigma_0, W, C10, K, direction):
    fiber_hom = constituent(rho_0, sigma_0, T_val, jnp.eye(3), W, 0.0, 0.0)
    fiber_gr = constituent(rho_0, sigma_0, T_val, jnp.eye(3), W, 0.1, 0.1)
    
    tau_history = [0.0]
    F_history_hom = [F_calc_uniax(1.0, direction)]
    F_history_gr = [F_calc_uniax(1.0, direction)]
    F_g_hom = [jnp.eye(3)]
    F_g_gr = [jnp.eye(3)]
    
    res_lam_hom, res_lam_gr = [1.0], [1.0]
    res_sig_hom, res_sig_gr = [sigma_0], [sigma_0]
    
    s = 0.0
    lam_h, lam_g = 1.0, 1.0
    sig_h, sig_g = sigma_0, sigma_0
    
    for step in range(1, n_steps + 1):
        s += ds
        tau_history.append(s)
        
        fiber_hom.append_converged_step(float(sig_h), ds)
        fiber_hom.m_history.append(fiber_hom.m_history[-1])
        fiber_hom.rho_history.append(fiber_hom.rho_history[-1])
        
        fiber_gr.append_converged_step(float(sig_g), ds)
        fiber_gr.m_history.append(fiber_gr.m_history[-1])
        fiber_gr.rho_history.append(fiber_gr.rho_history[-1])
        
        F_hist_h_arr = jnp.stack(F_history_hom)
        F_hist_g_arr = jnp.stack(F_history_gr)
        tau_hist_arr = jnp.array(tau_history)
        Fg_hist_h_arr = jnp.stack(F_g_hom)
        Fg_hist_g_arr = jnp.stack(F_g_gr)
        
        if case_type == "disp":
            lam_h, lam_g = lam_val, lam_val
            F_s_h, F_s_g = F_calc_uniax(lam_h, direction), F_calc_uniax(lam_g, direction)
            Fg_s_h, Fg_s_g = Fg_hist_h_arr[-1], Fg_hist_g_arr[-1]
            F_hist_h_temp = jnp.concatenate([F_hist_h_arr, F_s_h[None, ...]])
            F_hist_g_temp = jnp.concatenate([F_hist_g_arr, F_s_g[None, ...]])
            Fg_hist_h_temp = jnp.concatenate([Fg_hist_h_arr, Fg_s_h[None, ...]])
            Fg_hist_g_temp = jnp.concatenate([Fg_hist_g_arr, Fg_s_g[None, ...]])
            sig_h = get_stress_yy(lam_h, s, fiber_hom, F_hist_h_temp, tau_hist_arr, Fg_s_h, Fg_hist_h_temp, C10, K, direction)
            sig_g = get_stress_yy(lam_g, s, fiber_gr, F_hist_g_temp, tau_hist_arr, Fg_s_g, Fg_hist_g_temp, C10, K, direction)
        elif case_type == "stress":
            sig_h, sig_g = stress_val, stress_val
            Fg_s_h, Fg_s_g = Fg_hist_h_arr[-1], Fg_hist_g_arr[-1]
            lam_h = solve_lam(sig_h, lam_h, s, fiber_hom, F_hist_h_arr, tau_hist_arr, Fg_s_h, Fg_hist_h_arr, C10, K, direction)
            lam_g = solve_lam(sig_g, lam_g, s, fiber_gr, F_hist_g_arr, tau_hist_arr, Fg_s_g, Fg_hist_g_arr, C10, K, direction)
            F_s_h, F_s_g = F_calc_uniax(lam_h, direction), F_calc_uniax(lam_g, direction)
        else:
            lam_h, lam_g = lam_val[step], lam_val[step]
            F_s_h, F_s_g = F_calc_uniax(lam_h, direction), F_calc_uniax(lam_g, direction)
            Fg_s_h, Fg_s_g = Fg_hist_h_arr[-1], Fg_hist_g_arr[-1]
            F_hist_h_temp = jnp.concatenate([F_hist_h_arr, F_s_h[None, ...]])
            F_hist_g_temp = jnp.concatenate([F_hist_g_arr, F_s_g[None, ...]])
            Fg_hist_h_temp = jnp.concatenate([Fg_hist_h_arr, Fg_s_h[None, ...]])
            Fg_hist_g_temp = jnp.concatenate([Fg_hist_g_arr, Fg_s_g[None, ...]])
            sig_h = get_stress_yy(lam_h, s, fiber_hom, F_hist_h_temp, tau_hist_arr, Fg_s_h, Fg_hist_h_temp, C10, K, direction)
            sig_g = get_stress_yy(lam_g, s, fiber_gr, F_hist_g_temp, tau_hist_arr, Fg_s_g, Fg_hist_g_temp, C10, K, direction)
        
        F_history_hom.append(F_s_h)
        F_history_gr.append(F_s_g)
        
        rho_h = rho_calc(s, fiber_hom, tau_history, Q_calc, q_calc)
        fiber_hom.rho_history[-1] = float(rho_h)
        fiber_hom.m_history[-1] = float(m_calc(fiber_hom, rho_h, float(sig_h), sigma_0))
        F_g_hom.append((rho_h / rho_0)**(1/3) * jnp.eye(3))
        
        rho_g = rho_calc(s, fiber_gr, tau_history, Q_calc, q_calc)
        fiber_gr.rho_history[-1] = float(rho_g)
        fiber_gr.m_history[-1] = float(m_calc(fiber_gr, rho_g, float(sig_g), sigma_0))
        F_g_gr.append((rho_g / rho_0)**(1/3) * jnp.eye(3))
        
        res_lam_hom.append(float(lam_h))
        res_lam_gr.append(float(lam_g))
        res_sig_hom.append(float(sig_h))
        res_sig_gr.append(float(sig_g))

    if case_type == "disp":
        plot_results(tau_history, res_sig_hom, res_sig_gr, "Stress (MPa)", "Constant Deformation (Stress Relaxation)")
    elif case_type == "stress":
        plot_results(tau_history, res_lam_hom, res_lam_gr, "Stretch", "Constant Stress (Creep)")
    else:
        plot_results(tau_history, res_sig_hom, res_sig_gr, "Stress (MPa)", "Changing Deformation")

if __name__ == '__main__':
    c1, c2 = 10.0, 5.0
    C10, K = 0.0305, 0.610
    direction = (0.0, 1.0, 0.0)
    W_coll = fung_strain_energy(c1, c2, direction)
    
    simulate("disp", 1.5, None, 100, 10.0, 101.0, 1.0, 0.100, W_coll, C10, K, direction)
    simulate("stress", None, 0.120, 100, 10.0, 101.0, 1.0, 0.100, W_coll, C10, K, direction)
    
    dynamic_lams = [1.0 + 0.2 * jnp.sin(i / 10.0) for i in range(101)]
    simulate("dynamic", dynamic_lams, None, 100, 10.0, 101.0, 1.0, 0.100, W_coll, C10, K, direction)