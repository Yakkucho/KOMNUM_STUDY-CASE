import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. SETUP DATA (KELOMPOK 6 TUGAS 4)
# Fungsi Volume Air Sungai: V(t) = 500 + 80*cos(0.1t)
def V(t):
    return 500 + 80 * np.cos(0.1 * t)

# Data Diskrit (t=0 sampai t=14)
t_data = np.arange(0, 15, 1)
y_data = V(t_data)

# Data Halus (untuk kurva mulus)
t_smooth = np.linspace(0, 14, 200)
y_smooth = V(t_smooth)

# 2. PERHITUNGAN NUMERIK
# A. Interpolasi Spline (Menggambarkan kurva volume)
cs = CubicSpline(t_data, y_data)

# B. Turunan (Laju Perubahan Debit) using Central Difference
debit_change = np.gradient(y_data, t_data)

# C. Integral (Total Akumulasi Volume)
# PERBAIKAN 1: Ganti np.trapz menjadi np.trapezoid untuk NumPy terbaru
total_volume_time = np.trapezoid(y_data, t_data)

# 3. VISUALISASI GRAFIK
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# --- GRAFIK 1: VOLUME AIR (Interpolasi) ---
# PERBAIKAN 2: Tambahkan 'r' di depan string judul untuk format LaTeX
ax[0].set_title(r'1. Volume Air Sungai: $V(t) = 500 + 80\cos(0.1t)$', 
                fontsize=14, fontweight='bold')
ax[0].scatter(t_data, y_data, color='red', s=50, label='Data Pengamatan (Diskrit)', zorder=5)
ax[0].plot(t_smooth, cs(t_smooth), 'b', linewidth=2, label='Interpolasi Spline')
ax[0].set_ylabel('Volume Air ($m^3$)', fontsize=12)
ax[0].set_xlabel('Waktu (detik)', fontsize=10)
ax[0].legend(loc='upper right')
ax[0].grid(True, linestyle='--', alpha=0.6)

# --- GRAFIK 2: LAJU PERUBAHAN (Turunan) ---
ax[1].set_title('2. Laju Perubahan Volume (Turunan: $dV/dt$)', 
                fontsize=14, fontweight='bold')
ax[1].plot(t_data, debit_change, 'g-o', linewidth=2, label='Laju Perubahan ($m^3/s$)')
ax[1].axhline(0, color='black', linewidth=1.5)

# Warnai area (Hijau jika positif/naik, Merah jika negatif/surut)
ax[1].fill_between(t_data, debit_change, 0, where=(debit_change>0), 
                   color='green', alpha=0.1, label='Volume Naik (Pasang)')
ax[1].fill_between(t_data, debit_change, 0, where=(debit_change<0), 
                   color='red', alpha=0.1, label='Volume Turun (Surut)')

ax[1].set_ylabel('Laju Perubahan ($m^3/s$)', fontsize=12)
ax[1].set_xlabel('Waktu (detik)', fontsize=10)
ax[1].legend(loc='lower left')
ax[1].grid(True, linestyle='--', alpha=0.6)

# --- GRAFIK 3: INTEGRAL (Total Akumulasi) ---
ax[2].set_title(f'3. Akumulasi Ketersediaan Air (Integral Area: Total {total_volume_time:.2f})', 
                fontsize=14, fontweight='bold')
ax[2].plot(t_smooth, cs(t_smooth), 'k-', linewidth=1.5, label='Fungsi Volume')
ax[2].fill_between(t_smooth, cs(t_smooth), color='orange', alpha=0.4, label='Area Integral')

ax[2].set_ylabel('Volume Air ($m^3$)', fontsize=12)
ax[2].set_xlabel('Waktu (detik)', fontsize=10)
ax[2].legend(loc='upper right')
ax[2].grid(True, linestyle='--', alpha=0.6)
ax[2].set_ylim(0, 600) # Set limit Y dari 0 agar area terlihat jelas

plt.tight_layout(pad=3.0)
plt.show()