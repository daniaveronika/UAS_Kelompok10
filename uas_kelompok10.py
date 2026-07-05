# =========================================================
# PERBANDINGAN ARSITEKTUR CNN DAN LSTM UNTUK KLASIFIKASI
# TINGKAT KEPUASAN PEMBACA BERDASARKAN RATING BUKU
# PADA DATASET BOOK-CROSSING
# =========================================================

# ---------------------------------------------------------
# 1. IMPORT LIBRARY DAN INPUT DATASET
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("="*60)
print("1. INPUT DATASET")
print("="*60)

books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Books-Ratings.csv')
users = pd.read_csv('Users.csv')

print("Dataset Books:")
print(books.head())
print(f"\nInfo Dataset Books: {books.shape[0]} rows, {books.shape[1]} columns")

print("\nDataset Ratings:")
print(ratings.head())
print(f"\nInfo Dataset Ratings: {ratings.shape[0]} rows, {ratings.shape[1]} columns")

print("\nDataset Users:")
print(users.head())
print(f"\nInfo Dataset Users: {users.shape[0]} rows, {users.shape[1]} columns")

# Gabungkan ketiga dataset
df = pd.merge(books, ratings, on='ISBN')
df = pd.merge(df, users, on='User-ID')
print(f"\nData setelah digabung: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# 2. PREPROCESSING DATA
# ---------------------------------------------------------
print("\n" + "="*60)
print("2. PREPROCESSING DATA")
print("="*60)

# 2.1 Missing Value
print("\n" + "-"*50)
print("2.1 CEK & HAPUS MISSING VALUE")
print("-"*50)

print("\nMissing values sebelum preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nMengisi missing value Age dengan rata-rata: {df['Age'].mean():.2f}")
df['Age'] = df['Age'].fillna(df['Age'].mean())

df_before = df.shape[0]
df = df.dropna().reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows setelah dropna")

# 2.2 Outlier
print("\n" + "-"*50)
print("2.2 CEK & HAPUS OUTLIER")
print("-"*50)

print(f"\nMenghapus outlier Age (< 5 atau > 100 tahun)...")
df_before = df.shape[0]
df = df[(df['Age'] >= 5) & (df['Age'] <= 100)]
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

print(f"\nMenghapus rating tidak valid (di luar range 0-10)...")
df_before = df.shape[0]
df = df[(df['Book-Rating'] >= 0) & (df['Book-Rating'] <= 10)].reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

# Hanya gunakan rating eksplisit (rating > 0). Rating = 0 pada Book-Crossing
# berarti user hanya menambahkan buku tanpa memberi penilaian (implisit),
# sehingga tidak mencerminkan tingkat kepuasan pembaca yang sebenarnya.
print("\nMenghapus rating implisit (Book-Rating = 0)...")
df_before = df.shape[0]
df = df[df['Book-Rating'] > 0].reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

# 2.3 Normalisasi
print("\n" + "-"*50)
print("2.3 NORMALISASI")
print("-"*50)

user_ids = df['User-ID'].unique()
isbn_list = df['ISBN'].unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
isbn_to_idx = {b: i for i, b in enumerate(isbn_list)}
df['user_idx'] = df['User-ID'].map(user_to_idx)
df['isbn_idx'] = df['ISBN'].map(isbn_to_idx)
print(f"Total unique users: {len(user_ids)}")
print(f"Total unique books: {len(isbn_list)}")

# Simpan rating asli SEBELUM dinormalisasi (dipakai untuk membuat label kelas)
df['Book-Rating-Original'] = df['Book-Rating'].values

print("\nNormalisasi Book-Rating menggunakan MinMaxScaler...")
scaler = MinMaxScaler()
df['Book-Rating'] = scaler.fit_transform(df[['Book-Rating']])
print(f"Rating setelah normalisasi - Min: {df['Book-Rating'].min():.3f}, Max: {df['Book-Rating'].max():.3f}")

print("\nNormalisasi Age menggunakan MinMaxScaler...")
age_scaler = MinMaxScaler()
df['Age_normalized'] = age_scaler.fit_transform(df[['Age']])

print(f"\nDataset final setelah preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# 3. TRANSFORMATION
# ---------------------------------------------------------
print("\n" + "="*60)
print("3. TRANSFORMATION")
print("="*60)

# 3.1 Labeling -> Tingkat Kepuasan Pembaca (3 kelas)
print("\n" + "-"*50)
print("3.1 LABELING (Tingkat Kepuasan Pembaca)")
print("-"*50)

def tingkat_kepuasan(r):
    if r <= 4:
        return 0  # Tidak Puas
    elif r <= 7:
        return 1  # Cukup Puas
    else:
        return 2  # Sangat Puas

df['Kepuasan_Class'] = df['Book-Rating-Original'].apply(tingkat_kepuasan)
label_names = ['Tidak Puas (1-4)', 'Cukup Puas (5-7)', 'Sangat Puas (8-10)']
label_names_short = ['Tidak Puas', 'Cukup Puas', 'Sangat Puas']

print("\nDistribusi kelas tingkat kepuasan pembaca:")
class_dist = df['Kepuasan_Class'].value_counts().sort_index()
for i, cnt in class_dist.items():
    print(f"  {label_names[i]}: {cnt} data ({cnt/len(df)*100:.2f}%)")

plt.figure(figsize=(6, 4))
sns.barplot(x=label_names, y=class_dist.reindex(range(len(label_names)), fill_value=0).values,
            palette='viridis')
plt.title('Distribusi Tingkat Kepuasan Pembaca')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig('distribusi_kepuasan_pembaca.png', dpi=150)
plt.show()

# 3.2 Windowing -> Sliding window riwayat rating per user
print("\n" + "-"*50)
print("3.2 WINDOWING (Sliding Window Riwayat Rating per Pembaca)")
print("-"*50)

WINDOW_SIZE = 5
print(f"Ukuran window (jumlah rating historis yang dilihat): {WINDOW_SIZE}")
print("Catatan: dataset tidak memiliki timestamp, sehingga urutan kemunculan")
print("baris tiap user pada dataset digunakan sebagai proksi urutan waktu.")

X_seq, y_seq = [], []
used_users, skipped_users = 0, 0

for user_id, group in df.groupby('User-ID', sort=False):
    ratings_norm = group['Book-Rating'].values
    classes = group['Kepuasan_Class'].values

    if len(ratings_norm) < WINDOW_SIZE + 1:
        skipped_users += 1
        continue

    used_users += 1
    for i in range(len(ratings_norm) - WINDOW_SIZE):
        X_seq.append(ratings_norm[i:i + WINDOW_SIZE])
        y_seq.append(classes[i + WINDOW_SIZE])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"\nUser yang memenuhi syarat (>= {WINDOW_SIZE + 1} rating): {used_users}")
print(f"User yang dilewati (riwayat rating terlalu sedikit): {skipped_users}")
print(f"Total sampel hasil windowing: {X_seq.shape[0]}")

X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
print(f"Shape X akhir (samples, timesteps, features): {X_seq.shape}")
print(f"Shape y akhir: {y_seq.shape}")

# ---------------------------------------------------------
# 4. DATA SPLITTING (70% Train : 20% Validation : 10% Test)
# ---------------------------------------------------------
print("\n" + "="*60)
print("4. DATA SPLITTING")
print("="*60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.30, random_state=42, stratify=y_seq
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
)

print(f"Data Training   : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X_seq)*100:.1f}%)")
print(f"Data Validation : {X_val.shape[0]} sampel ({X_val.shape[0]/len(X_seq)*100:.1f}%)")
print(f"Data Testing    : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X_seq)*100:.1f}%)")

num_classes = len(label_names)

# ---------------------------------------------------------
# 5. KLASIFIKASI - METODE DEEP LEARNING 1: CNN (1D)
# ---------------------------------------------------------
print("\n" + "="*60)
print("5. KLASIFIKASI - METODE 1: CNN (1D Convolution)")
print("="*60)

model_cnn = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=2, activation='relu', padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()

print("\nMelatih model CNN...")
history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# ---------------------------------------------------------
# 5. KLASIFIKASI - METODE DEEP LEARNING 2: LSTM
# ---------------------------------------------------------
print("\n" + "="*60)
print("5. KLASIFIKASI - METODE 2: LSTM")
print("="*60)

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lstm.summary()

print("\nMelatih model LSTM...")
history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Grafik hasil training (accuracy & loss) CNN vs LSTM
print("\nMembuat grafik hasil training (accuracy & loss)...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(history_cnn.history['accuracy'], label='Train Accuracy')
axes[0, 0].plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_title('CNN - Model Accuracy'); axes[0, 0].legend()

axes[0, 1].plot(history_cnn.history['loss'], label='Train Loss')
axes[0, 1].plot(history_cnn.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('CNN - Model Loss'); axes[0, 1].legend()

axes[1, 0].plot(history_lstm.history['accuracy'], label='Train Accuracy')
axes[1, 0].plot(history_lstm.history['val_accuracy'], label='Validation Accuracy')
axes[1, 0].set_title('LSTM - Model Accuracy'); axes[1, 0].legend()

axes[1, 1].plot(history_lstm.history['loss'], label='Train Loss')
axes[1, 1].plot(history_lstm.history['val_loss'], label='Validation Loss')
axes[1, 1].set_title('LSTM - Model Loss'); axes[1, 1].legend()

plt.tight_layout()
plt.savefig('grafik_training_cnn_vs_lstm.png', dpi=150)
plt.show()

# ---------------------------------------------------------
# 6. EVALUASI & KOMPARASI
# ---------------------------------------------------------
print("\n" + "="*60)
print("6. EVALUASI & KOMPARASI (CNN vs LSTM)")
print("="*60)

def evaluasi_model(model, X_test, y_test, nama_model):
    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=label_names,
                                         zero_division=0, output_dict=True)

    print(f"\n--- Hasil Evaluasi Model {nama_model} ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\nClassification Report ({nama_model}):")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    return {'model': nama_model, 'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'cm': cm, 'y_pred': y_pred, 'report_dict': report_dict}

hasil_cnn = evaluasi_model(model_cnn, X_test, y_test, 'CNN')
hasil_lstm = evaluasi_model(model_lstm, X_test, y_test, 'LSTM')

# Confusion Matrix CNN vs LSTM
print("\nMembuat visualisasi confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(hasil_cnn['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
axes[0].set_title('Confusion Matrix - CNN')
axes[0].set_xlabel('Prediction'); axes[0].set_ylabel('Actual')

sns.heatmap(hasil_lstm['cm'], annot=True, fmt='d', cmap='Oranges',
            xticklabels=label_names, yticklabels=label_names, ax=axes[1])
axes[1].set_title('Confusion Matrix - LSTM')
axes[1].set_xlabel('Prediction'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix_cnn_vs_lstm.png', dpi=150)
plt.show()

# Bar Chart Perbandingan Metrik
print("\nMembuat bar chart perbandingan metrik CNN vs LSTM...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
cnn_scores = [hasil_cnn['accuracy'], hasil_cnn['precision'], hasil_cnn['recall'], hasil_cnn['f1']]
lstm_scores = [hasil_lstm['accuracy'], hasil_lstm['precision'], hasil_lstm['recall'], hasil_lstm['f1']]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width/2, cnn_scores, width, label='CNN', color='#4C72B0')
bars2 = ax.bar(x + width/2, lstm_scores, width, label='LSTM', color='#DD8452')

ax.set_ylabel('Skor')
ax.set_title('Perbandingan Performa CNN vs LSTM\n(Klasifikasi Tingkat Kepuasan Pembaca)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('bar_chart_perbandingan_cnn_lstm.png', dpi=150)
plt.show()

# Plot Hasil Testing (Aktual vs Prediksi CNN vs Prediksi LSTM)
N_TAMPIL = 30
idx_tampil = np.arange(min(N_TAMPIL, len(y_test)))

plt.figure(figsize=(11, 5))
plt.plot(idx_tampil, y_test[idx_tampil], marker='o', color='red', label='aktual')
plt.plot(idx_tampil, hasil_cnn['y_pred'][idx_tampil], marker='o', color='blue', label='CNN')
plt.plot(idx_tampil, hasil_lstm['y_pred'][idx_tampil], marker='o', color='green', label='LSTM')
plt.yticks([0, 1, 2], label_names)
plt.xlabel('Data ke-')
plt.ylabel('Tingkat Kepuasan')
plt.title('Plot Hasil Testing Tingkat Kepuasan Pembaca')
plt.legend()
plt.tight_layout()
plt.savefig('plot_hasil_testing_cnn_vs_lstm.png', dpi=150)
plt.show()

# Tabel Ringkasan & Model Terbaik
print("\n" + "-"*50)
print("RINGKASAN PERBANDINGAN CNN vs LSTM")
print("-"*50)

summary_df = pd.DataFrame({
    'Model': ['CNN', 'LSTM'],
    'Accuracy': [hasil_cnn['accuracy'], hasil_lstm['accuracy']],
    'Precision': [hasil_cnn['precision'], hasil_lstm['precision']],
    'Recall': [hasil_cnn['recall'], hasil_lstm['recall']],
    'F1-Score': [hasil_cnn['f1'], hasil_lstm['f1']]
})
print(summary_df.to_string(index=False))

best_idx = summary_df['F1-Score'].idxmax()
best_model_name = summary_df.loc[best_idx, 'Model']
print(f"\n>> Model terbaik berdasarkan F1-Score untuk klasifikasi tingkat kepuasan")
print(f"   pembaca adalah: {best_model_name}")