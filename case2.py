import numpy as np
import matplotlib.pyplot as plt

# Data diskrit
t = np.arange(0, 11, 2)
P = 50 * np.sin(0.2 * t) + 200

print("Data Tekanan:")
for ti, Pi in zip(t, P):
    print(f"t={ti:2.0f}   P(t)={Pi:.2f}")

# Interpolasi linear untuk t=5
P4 = P[2]  # t=4
P6 = P[3]  # t=6
P5_linear = P4 + (P6 - P4) * 0.5

P5_exact = 50 * np.sin(1.0) + 200

print(f"\nP(5) interpolasi: {P5_linear:.2f}")
print(f"P(5) eksak: {P5_exact:.2f}")
print(f"Selisih: {abs(P5_exact - P5_linear):.2f}")

# Turunan di t=4 (central difference)
P2 = P[1]
dPdt_4 = (P6 - P2) / 4

dPdt_exact = 10 * np.cos(0.8)

print(f"\nP'(4) numerik: {dPdt_4:.3f}")
print(f"P'(4) eksak: {dPdt_exact:.3f}")
print(f"Selisih: {abs(dPdt_exact - dPdt_4):.3f}")

# Integral dengan Simpson (0 sampai 10)
P0 = P[0]
P10 = P[-1]
I_simpson = (5/3) * (P0 + 4*P5_exact + P10)

I_exact = (-250 * np.cos(2.0) + 2000) - (-250 * np.cos(0) + 0)

print(f"\nIntegral Simpson: {I_simpson:.1f}")
print(f"Integral eksak: {I_exact:.1f}")
print(f"Selisih: {abs(I_simpson - I_exact):.1f}")

# Kurva halus
t_smooth = np.linspace(0, 10, 300)
P_smooth = np.interp(t_smooth, t, P)

# Plot 1
plt.figure(figsize=(9,5))
plt.plot(t_smooth, P_smooth, 'b-', linewidth=1.5)
plt.plot(t, P, 'ro', markersize=7)
plt.fill_between(t_smooth, P_smooth, alpha=0.2, color='green')
plt.xlabel("t")
plt.ylabel("P(t)")
plt.title("Tekanan vs Waktu")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Plot 2
plt.figure(figsize=(9,5))
turunan = 10 * np.cos(0.2 * t_smooth)
plt.plot(t_smooth, turunan, 'r-', linewidth=1.5)
plt.plot(4, dPdt_4, 'ko', markersize=8)
plt.xlabel("t")
plt.ylabel("P'(t)")
plt.title("Laju Perubahan Tekanan")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()