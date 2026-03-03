"""
Mehrdimensionaler Spline-Auswerter (vektorisiert)
Basiert auf: DOI 10.1007/s10614-007-9092-4, Abschnitt 3.2

Voraussetzungen:
  - _phi(t)       : Basisfunktion, vektorisiert, shape (...,)
  - _dphi_dt(t)   : 1. Ableitung
  - _d2phi_dt2(t) : 2. Ableitung
  - ci            : Koeffizientenarray, reshape zu (n1, n2, ..., nd) möglich
  - knots         : Liste von 1D-Arrays, knots[k] = Knotenvektor in Dimension k

Konvention:
  - x hat Shape (N, d): N Auswertungspunkte, d Dimensionen
  - Rückgabe: (f, grad, hess)
      f    : (N,)
      grad : (N, d)
      hess : (N, d, d)
"""

import numpy as np
from itertools import product as iproduct
# from .Spline1D import _phi, _dphi_dt, _d2phi_dt2


def _find_span_and_local(x_1d, knots_1d):
    """
    Für jeden Punkt in x_1d: finde den Knotenspan-Index und den
    lokalen Parameter t in [0, 1).

    Returns
    -------
    idx : (N,) int   – Index des linken Knotens im Span
    t   : (N,) float – lokaler Parameter
    """
    # Klemme Punkte an den gültigen Bereich
    knots_1d = np.asarray(knots_1d)
    x_c = np.clip(x_1d, knots_1d[0], knots_1d[-1])

    # Linker Knoten-Index (max. len-2 damit rechter Rand korrekt)
    idx = np.searchsorted(knots_1d, x_c, side='right') - 1
    idx = np.clip(idx, 0, len(knots_1d) - 2)

    h = knots_1d[idx + 1] - knots_1d[idx]
    t = (x_c - knots_1d[idx]) / h
    return idx, t, h


def eval_spline(x, ci, knots, shape):
    """
    Wertet den mehrdimensionalen Spline und seine Ableitungen aus.

    Parameters
    ----------
    x      : array (N, d)  – Auswertungspunkte
    ci     : array (n1*n2*...*nd,) oder schon in shape – Koeffizienten
    knots  : list of d arrays – Knotenvektoren je Dimension
    shape  : tuple (n1, n2, ..., nd) – Koeffizientenform

    Returns
    -------
    f    : (N,)      – Funktionswerte
    grad : (N, d)    – Gradient
    hess : (N, d, d) – Hessematrix
    """
    x = np.atleast_2d(x)
    N, d = x.shape
    C = np.reshape(ci, shape)

    # ------------------------------------------------------------------ #
    # 1) Für jede Dimension: aktive Indizes + phi, dphi, d2phi auswerten  #
    # ------------------------------------------------------------------ #
    # active_idx[k]  : (N, 4) int   – bis zu 4 aktive Basis-Indizes in Dim k
    # phi_vals[k]    : (N, 4) float
    # dphi_vals[k]   : (N, 4) float  (Ableitung nach t)
    # d2phi_vals[k]  : (N, 4) float

    active_idx = []
    phi_vals   = []
    dphi_vals  = []
    d2phi_vals = []
    h_arr      = []   # Knotenabstand (für Kettenregel)

    for k in range(d):
        knots_k = np.asarray(knots[k])
        span_idx, t_k, h_k = _find_span_and_local(x[:, k], knots_k)

        # n Stützstellen → n+2 Koeffizienten (je 1 extra links und rechts)
        # Koeff-Index für Span i: verschoben um +1 gegenüber span_idx
        # → aktive Indizes: span_idx+0, span_idx+1, span_idx+2, span_idx+3
        max_idx = shape[k] - 1
        offsets = np.array([0, 1, 2, 3])             # (4,)
        idx_mat = span_idx[:, None] + offsets[None, :]
        idx_mat = np.clip(idx_mat, 0, max_idx)

        # t_shifted: lokaler Parameter relativ zum jeweiligen Basisfunktionszentrum
        # Zentrum bei Koeff-Index span_idx+o → t = t_k - (o - 1) = t_k + 1 - o
        # (o=0 → t_k+1, o=1 → t_k, o=2 → t_k-1, o=3 → t_k-2)
        t_shifted = t_k[:, None] - (offsets[None, :] - 1)  # (N, 4)

        phi_mat   = _phi(t_shifted)        # (N, 4)
        dphi_mat  = _dphi_dt(t_shifted)    # (N, 4)
        d2phi_mat = _d2phi_dt2(t_shifted)  # (N, 4)

        active_idx.append(idx_mat)          # (N, 4)
        phi_vals.append(phi_mat)
        dphi_vals.append(dphi_mat / h_k[:, None])   # Kettenregel: dt/dx = 1/h
        d2phi_vals.append(d2phi_mat / h_k[:, None]**2)
        h_arr.append(h_k)

    # ------------------------------------------------------------------ #
    # 2) Über alle 4^d Kombinationen summieren                            #
    # ------------------------------------------------------------------ #
    f    = np.zeros(N)
    grad = np.zeros((N, d))
    hess = np.zeros((N, d, d))

    offsets_per_dim = [range(4)] * d   # je 4 Einträge pro Dimension

    for combo in iproduct(*offsets_per_dim):
        # combo = (o0, o1, ..., o_{d-1}), je ein lokaler Index 0..3

        # Koeffizientenindizes für alle N Punkte
        c_idx = tuple(active_idx[k][:, combo[k]] for k in range(d))  # d x (N,)
        c_vals = C[c_idx]   # (N,)

        # Produkt der Basisfunktionen (alle Dimensionen)
        phi_prod = np.ones(N)
        for k in range(d):
            phi_prod *= phi_vals[k][:, combo[k]]

        f += c_vals * phi_prod

        # Gradient: Produktregel – eine Dimension abgeleitet, Rest phi
        for j in range(d):
            dphi_prod = np.ones(N)
            for k in range(d):
                if k == j:
                    dphi_prod *= dphi_vals[k][:, combo[k]]
                else:
                    dphi_prod *= phi_vals[k][:, combo[k]]
            grad[:, j] += c_vals * dphi_prod

        # Hessematrix: zwei Dimensionen abgeleitet
        for j in range(d):
            for l in range(j, d):
                d2phi_prod = np.ones(N)
                for k in range(d):
                    if k == j and k == l:          # j == l: 2. Ableitung
                        d2phi_prod *= d2phi_vals[k][:, combo[k]]
                    elif k == j or k == l:         # gemischte Ableitung
                        d2phi_prod *= dphi_vals[k][:, combo[k]]
                    else:
                        d2phi_prod *= phi_vals[k][:, combo[k]]
                contrib = c_vals * d2phi_prod
                hess[:, j, l] += contrib
                if j != l:
                    hess[:, l, j] += contrib       # Symmetrie

    return f, grad, hess


# ======================================================================= #
#  Basisfunktionen gem. Gl. (2.4): Maximum bei t=0, Träger [-2, 2]        #
#                                                                          #
#        phi(t) = 1 - 5/2 t^2 + 3/2 |t|^3          für |t| < 1           #
#               = 2 - 4|t| + 5/2 t^2 - 1/2 |t|^3   für 1 <= |t| < 2     #
#               = 0                                  sonst                 #
#                                                                          #
#  (kubische Hermite-ähnliche Basis, symmetrisch in t)                    #
# ======================================================================= #


def _phi(t: float | np.ndarray) -> float | np.ndarray:
    t_abs = np.abs(t)
    result = np.zeros_like(t, dtype=float)
    
    # Case 1: 1 < |t| < 2
    mask1 = (t_abs > 1) & (t_abs <= 2)
    result[mask1] = (2 - t_abs[mask1])**3
    
    # Case 2: |t| < 1
    mask2 = t_abs <= 1
    result[mask2] = 4 - 6 * t_abs[mask2]**2 + 3 * t_abs[mask2]**3
    
    # Case 3: otherwise (already 0 from initialization)
    
    return result if isinstance(t, np.ndarray) else float(result)


def _dphi_dt(t: float | np.ndarray) -> float | np.ndarray:
    t_abs = np.abs(t)
    t_sign = np.sign(t)
    result = np.zeros_like(t, dtype=float)
    
    # Case 1: 1 < |t| < 2
    mask1 = (t_abs > 1) & (t_abs <= 2)
    result[mask1] = -t_sign[mask1]*3*(2 - t_abs[mask1])**2
    
    # Case 2: |t| < 1
    mask2 = t_abs <= 1
    result[mask2] =  - t_sign[mask2]*12 * t_abs[mask2] + t_sign[mask2]*9 * t_abs[mask2]**2
    
    # Case 3: otherwise (already 0 from initialization)
    
    return result if isinstance(t, np.ndarray) else float(result)


def _d2phi_dt2(t: float | np.ndarray) -> float | np.ndarray:
    t_abs = np.abs(t)
    result = np.zeros_like(t, dtype=float)
    
    # Case 1: 1 < |t| < 2
    mask1 = (t_abs > 1) & (t_abs <= 2)
    result[mask1] = 6*(2 - t_abs[mask1])
    
    # Case 2: |t| < 1
    mask2 = t_abs <= 1
    result[mask2] =  - 12 + 18 * t_abs[mask2]
    
    # Case 3: otherwise (already 0 from initialization)
    
    return result if isinstance(t, np.ndarray) else float(result)


# def _phi(t):
#     """
#     Basisfunktion nach Gl. (2.4): Maximum 1 bei t=0, Träger [-2, 2].
#     t : array beliebiger Shape
#     """
#     t = np.asarray(t, dtype=float)
#     at = np.abs(t)
#     result = np.zeros_like(t)

#     m1 = at < 1
#     result[m1] = 1.0 - 2.5 * at[m1]**2 + 1.5 * at[m1]**3

#     m2 = (at >= 1) & (at < 2)
#     result[m2] = 2.0 - 4.0*at[m2] + 2.5*at[m2]**2 - 0.5*at[m2]**3

#     return result


# def _dphi_dt(t):
#     """Erste Ableitung von _phi nach t."""
#     t = np.asarray(t, dtype=float)
#     at = np.abs(t)
#     sgn = np.sign(t)
#     result = np.zeros_like(t)

#     m1 = at < 1
#     result[m1] = sgn[m1] * (-5.0 * at[m1] + 4.5 * at[m1]**2)

#     m2 = (at >= 1) & (at < 2)
#     result[m2] = sgn[m2] * (-4.0 + 5.0*at[m2] - 1.5*at[m2]**2)

#     return result


# def _d2phi_dt2(t):
#     """Zweite Ableitung von _phi nach t."""
#     t = np.asarray(t, dtype=float)
#     at = np.abs(t)
#     result = np.zeros_like(t)

#     m1 = at < 1
#     result[m1] = -5.0 + 9.0 * at[m1]

#     m2 = (at >= 1) & (at < 2)
#     result[m2] = 5.0 - 3.0 * at[m2]

#     return result


# ======================================================================= #
#  Kleiner Selbsttest                                                       #
# ======================================================================= #
if __name__ == "__main__":
    # --- Konsistenzcheck _phi ---
    print("=== Konsistenzcheck _phi ===")
    print(f"phi(0)  = {_phi(np.array([0.0]))[0]:.6f}  (erwartet: 1.0)")
    print(f"phi(2)  = {_phi(np.array([2.0]))[0]:.6f}  (erwartet: 0.0)")
    print(f"phi(-2) = {_phi(np.array([-2.0]))[0]:.6f}  (erwartet: 0.0)")
    # Partition of Unity: sum_o phi(t - o) = 1 für t ∈ [0,1)
    t_test = np.linspace(0, 1, 200, endpoint=False)
    pou = sum(_phi(t_test - o) for o in [-1, 0, 1, 2])
    print(f"Partition of Unity max|sum-1| = {np.max(np.abs(pou - 1)):.2e}  (erwartet: ~0)")
    print()

    np.random.seed(0)

    # 2D-Beispiel: 5 Stützstellen → 7 Koeffizienten pro Richtung
    n_nodes = [5, 6]
    shape = tuple(n + 2 for n in n_nodes)            # (7, 8)
    ci = np.random.randn(np.prod(shape))
    knots = [np.linspace(0, 1, n_nodes[k]) for k in range(2)]

    # 3 Auswertungspunkte
    x = np.random.rand(3, 2) * 0.8 + 0.1   # sicher im Inneren

    f, grad, hess = eval_spline(x, ci, knots, shape)

    print("f    :", f)
    print("grad :", grad)
    print("hess :\n", hess)

    # Gradientencheck via finite differences
    eps = 1e-5
    grad_fd = np.zeros_like(grad)
    for j in range(2):
        xp, xm = x.copy(), x.copy()
        xp[:, j] += eps
        xm[:, j] -= eps
        fp, *_ = eval_spline(xp, ci, knots, shape)
        fm, *_ = eval_spline(xm, ci, knots, shape)
        grad_fd[:, j] = (fp - fm) / (2 * eps)

    print("\nGradient (analytisch):")
    print(grad)
    print("Gradient (FD):")
    print(grad_fd)
    print("Max. Fehler:", np.max(np.abs(grad - grad_fd)))
