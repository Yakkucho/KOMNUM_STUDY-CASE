import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import kagglehub

print("--- 1. Menyiapkan Data Temperature Change ---")
dataset_path = 'sevgisarac/temperature-change'
try:
    print(f"Mendownload dataset: {dataset_path}...")
    path = kagglehub.dataset_download(dataset_path)
    print("Lokasi file:", path)
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("Tidak ada file CSV ditemukan.")
    df_raw = pd.read_csv(os.path.join(path, csv_files[0]), encoding='latin1')
    print("Data berhasil dimuat!")
except Exception as e:
    print(f"Gagal download/load otomatis: {e}")
    exit()

cols = [str(c) for c in df_raw.columns]
if any('Y19' in c for c in cols):
    target_area = 'World' if 'World' in df_raw['Area'].values else df_raw['Area'].unique()[0]
    id_vars = [c for c in df_raw.columns if not c.startswith('Y')]
    df_melt = df_raw.melt(id_vars=id_vars, var_name='Year_Str', value_name='Temp_Change')
    df_melt['Year'] = df_melt['Year_Str'].str.replace('Y', '').astype(int)
    df_full = df_melt.groupby(['Area', 'Year'])['Temp_Change'].mean().reset_index()
    df_clean = df_full[df_full['Area'] == target_area].copy()
    area_name = target_area
else:
    df_full = df_raw.rename(columns={'Value': 'Temp_Change'})
    if 'Area' in df_full.columns:
        target_area = 'World' if 'World' in df_full['Area'].values else df_full['Area'].unique()[0]
        df_clean = df_full[df_full['Area'] == target_area].copy()
        area_name = target_area
    else:
        df_clean = df_full.copy()
        area_name = 'Global Sample'

df_task1 = df_clean[(df_clean['Year'] >= 1990) & (df_clean['Year'] <= 2020)].copy().dropna()
print(f"Dataset siap! Area: {area_name}, Total Data: {len(df_task1)} baris (1990-2020)")

print("\n--- 2. Membuat Grafik Insight Luas (Terpisah) ---")
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
plt.plot(df_task1['Year'], df_task1['Temp_Change'], marker='o', color='gray', alpha=0.5, label='Data Tahunan (Fluktuatif)')
df_task1['Rolling_Mean'] = df_task1['Temp_Change'].rolling(window=5).mean()
plt.plot(df_task1['Year'], df_task1['Rolling_Mean'], color='blue', linewidth=2.5, label='Rata-rata Bergerak 5 Tahun')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(f'Insight 1: Memisahkan Tren dari Fluktuasi Cuaca ({area_name})', fontsize=14)
plt.ylabel('Anomali Suhu (°C)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('1_Insight_Tren_Rolling.png')
print("Grafik 1 disimpan: '1_Insight_Tren_Rolling.png'")

plt.figure(figsize=(10, 6))
df_task1['Decade'] = (df_task1['Year'] // 10) * 10
decade_data = df_task1.groupby('Decade')['Temp_Change'].mean().reset_index()
decade_data['Decade_Label'] = decade_data['Decade'].astype(str) + 's'
sns.barplot(x='Decade_Label', y='Temp_Change', data=decade_data, palette='Reds')
plt.title('Insight 2: Kenaikan Suhu Rata-rata per Dekade', fontsize=14)
plt.ylabel('Rata-rata Anomali (°C)', fontsize=12)
plt.xlabel('Dekade', fontsize=12)
for index, row in decade_data.iterrows():
    plt.text(index, row.Temp_Change, f"{row.Temp_Change:.2f}°C", color='black', ha="center", va="bottom")
plt.tight_layout()
plt.savefig('2_Insight_Perbandingan_Dekade.png')
print("Grafik 2 disimpan: '2_Insight_Perbandingan_Dekade.png'")

plt.figure(figsize=(10, 6))
sns.histplot(df_task1['Temp_Change'], kde=True, color='orange', bins=10)
plt.axvline(0, color='black', linestyle='--', label='Titik 0 (Referensi Normal)')
plt.axvline(df_task1['Temp_Change'].mean(), color='red', linestyle='-', label=f'Rata-rata Saat Ini ({df_task1["Temp_Change"].mean():.2f}°C)')
plt.title('Insight 3: Pergeseran "Kenormalan Baru" (New Normal)', fontsize=14)
plt.xlabel('Anomali Suhu (°C)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('3_Insight_Distribusi_Suhu.png')
print("Grafik 3 disimpan: '3_Insight_Distribusi_Suhu.png'")

print("\n--- 3. Membuat Model Numerik & Grafik Utama ---")
X_original = df_task1['Year'].values
X = X_original - X_original.min()
Y = df_task1['Temp_Change'].values
coeffs = np.polyfit(X, Y, 2)
model_func = np.poly1d(coeffs)
equation_str = f"T(t) = {coeffs[0]:.5f}t² + {coeffs[1]:.4f}t + {coeffs[2]:.4f}"
Y_pred = model_func(X)
r2 = r2_score(Y, Y_pred)
print(f"Model: {equation_str}")
print(f"R² Score: {r2:.4f}")

comparison_df = pd.DataFrame({'Tahun': X_original, 't': X, 'Suhu_Aktual': Y, 'Prediksi_Model': np.round(Y_pred, 3), 'Error': np.round(Y - Y_pred, 3)})
comparison_df.to_csv('4_Data_Perbandingan_Suhu.csv', index=False)
print("Data Excel disimpan: '4_Data_Perbandingan_Suhu.csv'")

plt.figure(figsize=(12, 7))
plt.scatter(X_original, Y, color='#1f77b4', s=80, label='Data Aktual', zorder=2)
X_smooth = np.linspace(X.min(), X.max(), 100)
Y_smooth = model_func(X_smooth)
plt.plot(X_smooth + X_original.min(), Y_smooth, color='#d62728', linewidth=3, label='Model Kuadratik (Numerik)')
for i in range(len(X)):
    plt.plot([X_original[i], X_original[i]], [Y[i], Y_pred[i]], color='gray', linestyle=':', alpha=0.5)
plt.text(X_original.min(), Y.max(), f"Model: {equation_str}\n(t=0 pada tahun 1990)", fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round'))
plt.title(f'FINAL MODEL: Akselerasi Pemanasan Global (Task 1)\nArea: {area_name}', fontsize=16)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Perubahan Suhu (°C)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('4_Final_Model_Matematika.png')
print("Grafik Model Final disimpan: '4_Final_Model_Matematika.png'")

print("\n--- 4. Analisis Khusus Indonesia ---")
target_country = 'Indonesia'
df_indo = pd.DataFrame()
if 'Area' in df_full.columns:
    df_indo = df_full[df_full['Area'] == target_country].copy().dropna(subset=['Temp_Change'])

if not df_indo.empty:
    print(f"Data {target_country} ditemukan! Melakukan analisis spesifik...")
    df_indo_task = df_indo[(df_indo['Year'] >= 1990) & (df_indo['Year'] <= 2020)].copy()
    if len(df_indo_task) > 5:
        X_indo = df_indo_task['Year'].values
        X_indo_norm = X_indo - X_indo.min()
        Y_indo = df_indo_task['Temp_Change'].values
        coeffs_indo = np.polyfit(X_indo_norm, Y_indo, 2)
        model_indo = np.poly1d(coeffs_indo)
        eq_indo = f"T(t) = {coeffs_indo[0]:.5f}t² + {coeffs_indo[1]:.4f}t + {coeffs_indo[2]:.4f}"
        
        plt.figure(figsize=(12, 7))
        plt.scatter(X_indo, Y_indo, color='green', s=80, label='Data Aktual (Indonesia)', zorder=2)
        X_smooth = np.linspace(X_indo_norm.min(), X_indo_norm.max(), 100)
        Y_smooth = model_indo(X_smooth)
        plt.plot(X_smooth + X_indo.min(), Y_smooth, color='orange', linewidth=3, label='Model Kuadratik (Indonesia)')
        plt.title(f'Analisis Khusus: Kenaikan Suhu di {target_country}', fontsize=16)
        plt.xlabel('Tahun', fontsize=12)
        plt.ylabel('Perubahan Suhu (°C)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.text(X_indo.min(), Y_indo.max(), f"Model Indo: {eq_indo}", fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', boxstyle='round'))
        plt.savefig('5_Analisis_Indonesia.png')
        print("Grafik Indonesia disimpan: '5_Analisis_Indonesia.png'")
        
        df_indo_out = pd.DataFrame({'Tahun': X_indo, 'Suhu_Indo': Y_indo, 'Model_Indo': np.round(model_indo(X_indo_norm), 3)})
        df_indo_out.to_csv('5_Data_Indonesia.csv', index=False)
        print("Data Indonesia disimpan: '5_Data_Indonesia.csv'")
    else:
        print("Data Indonesia terlalu sedikit untuk dianalisis (kurang dari 5 titik data antara 1990-2020).")
else:
    print("Data 'Indonesia' tidak ditemukan dalam dataset ini.")

print("\n--- SELESAI ---")