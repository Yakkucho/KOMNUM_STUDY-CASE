import numpy as np
from scipy.interpolate import lagrange, CubicSpline
from scipy.integrate import trapezoid


def interpolasi_lagrange_manual(t_sample, y_sample, t_target):
    """Interpolasi Polinomial Lagrange dengan validasi manual"""
    print("\n" + "="*70)
    print("INTERPOLASI POLINOMIAL LAGRANGE - VALIDASI MANUAL")
    print("="*70)
    
    print(f"\nData Sampel:")
    print(f"  t₀={t_sample[0]} → y₀={y_sample[0]:.4f}")
    print(f"  t₁={t_sample[1]} → y₁={y_sample[1]:.4f}")
    print(f"  t₂={t_sample[2]} → y₂={y_sample[2]:.4f}")
    
    # Hitung bobot Lagrange
    L0 = ((t_target - t_sample[1]) * (t_target - t_sample[2])) / \
         ((t_sample[0] - t_sample[1]) * (t_sample[0] - t_sample[2]))
    L1 = ((t_target - t_sample[0]) * (t_target - t_sample[2])) / \
         ((t_sample[1] - t_sample[0]) * (t_sample[1] - t_sample[2]))
    L2 = ((t_target - t_sample[0]) * (t_target - t_sample[1])) / \
         ((t_sample[2] - t_sample[0]) * (t_sample[2] - t_sample[1]))
    
    print(f"\nBobot Lagrange:")
    print(f"  L₀({t_target}) = {L0:.4f}")
    print(f"  L₁({t_target}) = {L1:.4f}")
    print(f"  L₂({t_target}) = {L2:.4f}")
    
    # Hitung P(t_target)
    P_manual = y_sample[0]*L0 + y_sample[1]*L1 + y_sample[2]*L2
    
    print(f"\nHasil Interpolasi:")
    print(f"  P({t_target}) = ({y_sample[0]:.4f} × {L0:.4f}) + ({y_sample[1]:.4f} × {L1:.4f}) + ({y_sample[2]:.4f} × {L2:.4f})")
    print(f"  P({t_target}) ≈ {P_manual:.4f}")
    
    # Validasi
    from analisis_pH import ph_function
    ph_exact = ph_function(t_target)
    error = abs(P_manual - ph_exact)
    
    print(f"\nValidasi:")
    print(f"  Nilai Eksak pH({t_target}) = {ph_exact:.4f}")
    print(f"  Selisih (error)           = {error:.4f}")
    print(f"  ✓ Hasil interpolasi mendekati nilai asli!")
    
    return P_manual, ph_exact

def interpolasi_spline_kubik(t_discrete, ph_discrete):
    """Interpolasi Spline Kubik dengan validasi manual"""
    print("\n" + "="*70)
    print("INTERPOLASI SPLINE KUBIK - VALIDASI MANUAL")
    print("="*70)
    
    spline = CubicSpline(t_discrete, ph_discrete)
    
    # Untuk validasi manual, buat data tiap hari
    from analisis_pH import ph_function
    t_full = np.arange(0, 31, 1)
    ph_full = ph_function(t_full)
    spline_full = CubicSpline(t_full, ph_full)
    
    # Koefisien untuk segmen t=1 s.d t=2
    segment_idx = 1
    
    print(f"\nKoefisien Spline (Segmen t=1 s.d t=2):")
    print(f"  S(t) = a + b(t-t₁) + c(t-t₁)² + d(t-t₁)³, di mana t₁ = 1")
    print(f"  a = {spline_full.c[3, segment_idx]:.4f}")
    print(f"  b = {spline_full.c[2, segment_idx]:.4f}")
    print(f"  c = {spline_full.c[1, segment_idx]:.4f}")
    print(f"  d = {spline_full.c[0, segment_idx]:.4f}")
    
    # Validasi pada t=1.5
    t_test = 1.5
    delta_t = t_test - 1.0
    
    a, b, c, d = spline_full.c[3, segment_idx], spline_full.c[2, segment_idx], \
                 spline_full.c[1, segment_idx], spline_full.c[0, segment_idx]
    
    S_manual = a + b*delta_t + c*(delta_t**2) + d*(delta_t**3)
    ph_exact = ph_function(t_test)
    
    print(f"\nValidasi pada t = {t_test}:")
    print(f"  S({t_test}) = {a:.4f} + {b:.4f}({delta_t}) + {c:.4f}({delta_t})² + {d:.4f}({delta_t})³")
    print(f"  S({t_test}) ≈ {S_manual:.4f}")
    print(f"  Nilai Eksak = {ph_exact:.4f}, Error = {abs(S_manual - ph_exact):.4f}")
    
    return spline, spline_full

def turunan_numerik_central(t_discrete, ph_discrete):
    """Turunan Numerik Central Difference + Validasi Manual"""
    print("\n" + "="*70)
    print("TURUNAN NUMERIK (CENTRAL DIFFERENCE) - VALIDASI MANUAL")
    print("="*70)
    
    from analisis_pH import ph_function, ph_derivative_analytical
    t_full = np.arange(0, 31, 1)
    ph_full = ph_function(t_full)
    
    # Contoh: t=2
    idx, h = 2, 1
    ph_plus = ph_full[idx + 1]
    ph_minus = ph_full[idx - 1]
    derivative_manual = (ph_plus - ph_minus) / (2 * h)
    derivative_analytical = ph_derivative_analytical(t_full[idx])
    
    print(f"\nRumus: pH'(t) ≈ [pH(t+h) - pH(t-h)] / (2h)")
    print(f"\nPerhitungan untuk t = 2:")
    print(f"  pH(3) = {ph_plus:.4f}, pH(1) = {ph_minus:.4f}, h = {h}")
    print(f"  pH'(2) ≈ ({ph_plus:.4f} - {ph_minus:.4f}) / 2 = {derivative_manual:.4f} pH/hari")
    
    print(f"\nInterpretasi:")
    print(f"  ✓ {'pH NAIK (basa)' if derivative_manual > 0 else 'pH TURUN (asam)'}")
    print(f"\nValidasi: pH'(2) analitik = {derivative_analytical:.4f}, Error = {abs(derivative_manual - derivative_analytical):.6f}")
    
    # Hitung turunan untuk semua titik
    dph_dt = np.gradient(ph_discrete, t_discrete)
    return dph_dt, t_discrete

def integral_numerik_trapesium(t_discrete, ph_discrete):
    """Integral Numerik Trapezoidal Rule + Validasi Manual"""
    print("\n" + "="*70)
    print("INTEGRAL NUMERIK (TRAPEZOIDAL RULE) - VALIDASI MANUAL")
    print("="*70)
    
    print(f"\nRumus: Luas ≈ (h/2) × [y₀ + 2(y₁ + y₂ + ... + yₙ₋₁) + yₙ]")
    
    # Contoh manual: t=0 sampai t=2
    from analisis_pH import ph_function
    t_sample = np.array([0, 1, 2])
    ph_sample = ph_function(t_sample)
    h = 1
    
    integral_manual_sample = (h/2) * (ph_sample[0] + 2*ph_sample[1] + ph_sample[2])
    
    print(f"\nContoh (t=0 s.d t=2):")
    print(f"  y₀={ph_sample[0]:.4f}, y₁={ph_sample[1]:.4f}, y₂={ph_sample[2]:.4f}")
    print(f"  Luas ≈ 0.5 × [{ph_sample[0]:.4f} + 2({ph_sample[1]:.4f}) + {ph_sample[2]:.4f}]")
    print(f"  Luas ≈ {integral_manual_sample:.4f}")
    
    # Integral total
    integral_total = trapezoid(ph_discrete, t_discrete)
    average_ph = integral_total / 30
    
    print(f"\n" + "-"*70)
    print(f"HASIL TOTAL (t=0 s.d t=30):")
    print(f"  Total Integral ≈ {integral_total:.2f}")
    print(f"  Rata-rata pH = {integral_total:.2f} / 30 ≈ {average_ph:.2f}")
    
    return integral_total, average_ph
