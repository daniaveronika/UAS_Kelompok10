# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                      Flatten, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Random seed supaya hasil training konsisten
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Input dataset
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

df = pd.merge(books, ratings, on='ISBN')
df = pd.merge(df, users, on='User-ID')
print(f"\nData setelah digabung: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Preprocessing Data
print("\n" + "="*60)
print("2. PREPROCESSING DATA")
print("="*60)

# 2.1 Cek dan Hapus Missing Value
print("\n" + "-"*50)
print("2.1 CEK & HAPUS MISSING VALUE")
print("-"*50)

print("\nMissing values sebelum preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nMengisi missing value Age dengan rata-rata: {df['Age'].mean():.2f}")
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Book-Author'] = df['Book-Author'].fillna('Unknown')
df['Publisher'] = df['Publisher'].fillna('Unknown')

df_before = df.shape[0]
df = df.dropna(subset=['Book-Rating', 'ISBN', 'User-ID']).reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows setelah dropna")

# 2.2 Cek dan Hapus Outlier
print("\n" + "-"*50)
print("2.2 CEK & HAPUS OUTLIER")
print("-"*50)

print("\nMenghapus outlier Age (< 5 atau > 100 tahun)...")
df_before = df.shape[0]
df = df[(df['Age'] >= 5) & (df['Age'] <= 100)]
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

print("\nMenghapus rating tidak valid (di luar range 0-10)...")
df_before = df.shape[0]
df = df[(df['Book-Rating'] >= 0) & (df['Book-Rating'] <= 10)].reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

print("\nMenghapus rating implisit (Book-Rating = 0)...")
df_before = df.shape[0]
df = df[df['Book-Rating'] > 0].reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

print("\nMenghapus outlier Year-Of-Publication (< 1900 atau > 2026)...")
df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
df['Year-Of-Publication'] = df['Year-Of-Publication'].fillna(df['Year-Of-Publication'].median())
df_before = df.shape[0]
df = df[(df['Year-Of-Publication'] >= 1900) & (df['Year-Of-Publication'] <= 2026)].reset_index(drop=True)
print(f"Data berkurang dari {df_before} menjadi {df.shape[0]} rows")

# 3. Transformation
print("\n" + "="*60)
print("3. TRANSFORMATION")
print("="*60)

# 3.1 Normalisasi
print("\n" + "-"*50)
print("3.1 NORMALISASI")
print("-"*50)

age_scaler = MinMaxScaler()
df['Age_normalized'] = age_scaler.fit_transform(df[['Age']])

year_scaler = MinMaxScaler()
df['Year_normalized'] = year_scaler.fit_transform(df[['Year-Of-Publication']])

print(f"Jumlah author unik    : {df['Book-Author'].nunique()}")
print(f"Jumlah publisher unik : {df['Publisher'].nunique()}")

# 3.2 Labeling
print("\n" + "-"*50)
print("3.2 LABELING (Suka vs Tidak Suka)")
print("-"*50)

def suka_atau_tidak(r):
    return 1 if r >= 8 else 0  # 1 = Suka, 0 = Tidak Suka

df['Suka_Class'] = df['Book-Rating'].apply(suka_atau_tidak)
label_names = ['Tidak Suka (<8)', 'Suka (>=8)']
label_names_short = ['Tidak Suka', 'Suka']

print("\nDistribusi kelas Suka vs Tidak Suka:")
class_dist = df['Suka_Class'].value_counts().sort_index()
for i, cnt in class_dist.items():
    print(f"  {label_names[i]}: {cnt} data ({cnt/len(df)*100:.2f}%)")

plt.figure(figsize=(6, 4))
sns.barplot(x=label_names, y=class_dist.reindex(range(len(label_names)), fill_value=0).values,
            palette='viridis')
plt.title('Distribusi Label Suka vs Tidak Suka')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# 4. Data Splitting (70% Train : 20% Validation : 10% Test)
print("\n" + "="*60)
print("4. DATA SPLITTING")
print("="*60)

idx_train, idx_temp = train_test_split(
    df.index, test_size=0.30, random_state=SEED, stratify=df['Suka_Class']
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=(1/3), random_state=SEED, stratify=df.loc[idx_temp, 'Suka_Class']
)

df['split_set'] = 'train'
df.loc[idx_val, 'split_set'] = 'val'
df.loc[idx_test, 'split_set'] = 'test'

print(f"Data Training   : {(df['split_set']=='train').sum()} sampel "
      f"({(df['split_set']=='train').sum()/len(df)*100:.1f}%)")
print(f"Data Validation : {(df['split_set']=='val').sum()} sampel "
      f"({(df['split_set']=='val').sum()/len(df)*100:.1f}%)")
print(f"Data Testing    : {(df['split_set']=='test').sum()} sampel "
      f"({(df['split_set']=='test').sum()/len(df)*100:.1f}%)")

train_mask = df['split_set'] == 'train'
val_mask = df['split_set'] == 'val'
test_mask = df['split_set'] == 'test'

train_df = df[train_mask].copy()
val_df = df[val_mask].copy()
test_df = df[test_mask].copy()

# 3.3 Fitur Perilaku: prev & dist (lanjutan Transformation, dari Training set setelah splitting)
print("\n" + "-"*50)
print("3.3 FITUR PERILAKU (prev & dist) -- dibangun hanya dari Training set")
print("-"*50)
print("prev : indikator apakah rating sebelumnya dari user tersebut positif (Suka)")
print("dist : rata-rata berjalan proporsi rating positif dari user tersebut sebelumnya")

train_df['prev'] = train_df.groupby('User-ID')['Suka_Class'].shift(1).fillna(0.5)
train_df['dist'] = (
    train_df.groupby('User-ID')['Suka_Class']
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
global_mean_train = train_df['Suka_Class'].mean()
train_df['dist'] = train_df['dist'].fillna(global_mean_train)

user_stats = train_df.groupby('User-ID')['Suka_Class'].agg(sum='sum', count='count')
user_stats['dist_final'] = (user_stats['sum'] + 1) / (user_stats['count'] + 2)
user_stats['prev_final'] = train_df.groupby('User-ID')['Suka_Class'].last()

for part_df in (val_df, test_df):
    part_df['prev'] = part_df['User-ID'].map(user_stats['prev_final']).fillna(0.5)
    part_df['dist'] = part_df['User-ID'].map(user_stats['dist_final']).fillna(global_mean_train)

print(f"\nContoh nilai prev (Training) : {train_df['prev'].head().tolist()}")
print(f"Contoh nilai dist (Training) : {train_df['dist'].round(3).head().tolist()}")
print(f"Jumlah user unik di Training yang punya statistik prev/dist: {len(user_stats)}")

# 3.4 Target Encoding Author & Publisher (leave-one-out, hanya Training)
print("\n" + "-"*50)
print("3.4 TARGET ENCODING AUTHOR & PUBLISHER (hanya dari Training)")
print("-"*50)
print("Menggunakan leave-one-out: label baris itu sendiri dikeluarkan dari")
print("perhitungan rata-ratanya sendiri, untuk mencegah data leakage.")

GLOBAL_SMOOTH_K = 10

author_stats = train_df.groupby('Book-Author')['Suka_Class'].agg(sum='sum', count='count')
author_stats['Author_TargetEnc'] = (
    (author_stats['sum'] + GLOBAL_SMOOTH_K * global_mean_train) /
    (author_stats['count'] + GLOBAL_SMOOTH_K)
)

publisher_stats = train_df.groupby('Publisher')['Suka_Class'].agg(sum='sum', count='count')
publisher_stats['Publisher_TargetEnc'] = (
    (publisher_stats['sum'] + GLOBAL_SMOOTH_K * global_mean_train) /
    (publisher_stats['count'] + GLOBAL_SMOOTH_K)
)

author_sum_map = train_df.groupby('Book-Author')['Suka_Class'].transform('sum')
author_count_map = train_df.groupby('Book-Author')['Suka_Class'].transform('count')
train_df['Author_TargetEnc'] = (
    (author_sum_map - train_df['Suka_Class'] + GLOBAL_SMOOTH_K * global_mean_train) /
    (author_count_map - 1 + GLOBAL_SMOOTH_K)
)

publisher_sum_map = train_df.groupby('Publisher')['Suka_Class'].transform('sum')
publisher_count_map = train_df.groupby('Publisher')['Suka_Class'].transform('count')
train_df['Publisher_TargetEnc'] = (
    (publisher_sum_map - train_df['Suka_Class'] + GLOBAL_SMOOTH_K * global_mean_train) /
    (publisher_count_map - 1 + GLOBAL_SMOOTH_K)
)

for part_df in (val_df, test_df):
    part_df['Author_TargetEnc'] = (
        part_df['Book-Author'].map(author_stats['Author_TargetEnc']).fillna(global_mean_train)
    )
    part_df['Publisher_TargetEnc'] = (
        part_df['Publisher'].map(publisher_stats['Publisher_TargetEnc']).fillna(global_mean_train)
    )

print(f"\nJumlah author unik di Training   : {len(author_stats)}")
print(f"Jumlah publisher unik di Training : {len(publisher_stats)}")
print(f"Contoh Author_TargetEnc (Training, leave-one-out): "
      f"{train_df['Author_TargetEnc'].round(3).head().tolist()}")

# 3.5 Target Encoding ISBN + Popularitas Buku (leave-one-out, hanya Training)
print("\n" + "-"*50)
print("3.5 TARGET ENCODING ISBN + POPULARITAS BUKU (hanya dari Training)")
print("-"*50)

ISBN_SMOOTH_K = 5

isbn_stats = train_df.groupby('ISBN')['Suka_Class'].agg(sum='sum', count='count')
isbn_stats['ISBN_TargetEnc'] = (
    (isbn_stats['sum'] + ISBN_SMOOTH_K * global_mean_train) /
    (isbn_stats['count'] + ISBN_SMOOTH_K)
)
isbn_stats['ISBN_LogCount'] = np.log1p(isbn_stats['count'])
max_log_count = isbn_stats['ISBN_LogCount'].max()
isbn_stats['ISBN_LogCount_norm'] = isbn_stats['ISBN_LogCount'] / max_log_count

isbn_sum_map = train_df.groupby('ISBN')['Suka_Class'].transform('sum')
isbn_count_map = train_df.groupby('ISBN')['Suka_Class'].transform('count')
train_df['ISBN_TargetEnc'] = (
    (isbn_sum_map - train_df['Suka_Class'] + ISBN_SMOOTH_K * global_mean_train) /
    (isbn_count_map - 1 + ISBN_SMOOTH_K)
)
train_df['ISBN_Popularitas'] = train_df['ISBN'].map(isbn_stats['ISBN_LogCount_norm'])

for part_df in (val_df, test_df):
    part_df['ISBN_TargetEnc'] = (
        part_df['ISBN'].map(isbn_stats['ISBN_TargetEnc']).fillna(global_mean_train)
    )
    part_df['ISBN_Popularitas'] = (
        part_df['ISBN'].map(isbn_stats['ISBN_LogCount_norm']).fillna(0.0)
    )

print(f"\nJumlah ISBN unik di Training: {len(isbn_stats)}")
print(f"Contoh ISBN_TargetEnc (Training, leave-one-out): "
      f"{train_df['ISBN_TargetEnc'].round(3).head().tolist()}")
print(f"Contoh ISBN_Popularitas (Training): {train_df['ISBN_Popularitas'].round(3).head().tolist()}")

df = pd.concat([train_df, val_df, test_df]).sort_index()

# 3.6 Menyusun Fitur Akhir
print("\n" + "-"*50)
print("3.6 MENYUSUN FITUR PER PASANGAN USER-BUKU")
print("-"*50)

fitur_kolom = ['Age_normalized', 'Author_TargetEnc', 'Publisher_TargetEnc',
               'Year_normalized', 'prev', 'dist',
               'ISBN_TargetEnc', 'ISBN_Popularitas']
NUM_FITUR = len(fitur_kolom)

train_mask = df['split_set'] == 'train'
val_mask = df['split_set'] == 'val'
test_mask = df['split_set'] == 'test'

X_train = df.loc[train_mask, fitur_kolom].values.reshape(-1, NUM_FITUR, 1)
X_val = df.loc[val_mask, fitur_kolom].values.reshape(-1, NUM_FITUR, 1)
X_test = df.loc[test_mask, fitur_kolom].values.reshape(-1, NUM_FITUR, 1)

y_train = df.loc[train_mask, 'Suka_Class'].values
y_val = df.loc[val_mask, 'Suka_Class'].values
y_test = df.loc[test_mask, 'Suka_Class'].values

print(f"\nFitur yang digunakan: {fitur_kolom}")
print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_val   : {X_val.shape}")
print(f"Shape X_test  : {X_test.shape}")

bobot_kelas = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: bobot_kelas[i] for i in range(len(bobot_kelas))}
print(f"\nClass weight yang digunakan (mengatasi ketidakseimbangan kelas): {class_weight_dict}")

# 5. Klasifikasi - Metode Deep Learning 1: CNN (1D)
print("\n" + "="*60)
print("5. KLASIFIKASI - METODE 1: CNN (1D Convolution)")
print("="*60)

L2_REG = 1e-4

model_cnn = Sequential([
    Conv1D(48, kernel_size=2, activation='relu', kernel_regularizer=l2(L2_REG),
           input_shape=(NUM_FITUR, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(24, kernel_size=2, activation='relu', kernel_regularizer=l2(L2_REG),
           padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.summary()

print("\nMelatih model CNN...")
history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ],
    verbose=1
)

# 5. Klasifikasi - Metode Deep Learning 2: LSTM
print("\n" + "="*60)
print("5. KLASIFIKASI - METODE 2: LSTM")
print("="*60)

model_lstm = Sequential([
    LSTM(48, return_sequences=True, kernel_regularizer=l2(L2_REG),
         input_shape=(NUM_FITUR, 1)),
    BatchNormalization(),
    LSTM(24, kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()

print("\nMelatih model LSTM...")
history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=128,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ],
    verbose=1
)

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
plt.show()

# 6. Evaluasi
print("\n" + "="*60)
print("6. EVALUASI & KOMPARASI (CNN vs LSTM)")
print("="*60)

# 6.1 Menghitung hasil evaluasi model
def evaluasi_model(model, X_test, y_test, nama_model):
    prob = model.predict(X_test).ravel()
    y_pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
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

# 6.2 Confusion Matrix & Bar Chart Perbandingan Metrik
print("\nMembuat visualisasi confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

sns.heatmap(hasil_cnn['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
axes[0].set_title('Confusion Matrix - CNN')
axes[0].set_xlabel('Prediction'); axes[0].set_ylabel('Actual')

sns.heatmap(hasil_lstm['cm'], annot=True, fmt='d', cmap='Oranges',
            xticklabels=label_names, yticklabels=label_names, ax=axes[1])
axes[1].set_title('Confusion Matrix - LSTM')
axes[1].set_xlabel('Prediction'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

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
ax.set_title('Perbandingan Performa CNN vs LSTM\n(Klasifikasi Suka vs Tidak Suka)')
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
plt.show()

N_TAMPIL = 30
idx_tampil = np.arange(min(N_TAMPIL, len(y_test)))

plt.figure(figsize=(11, 5))
plt.plot(idx_tampil, y_test[idx_tampil], marker='o', color='red', label='aktual')
plt.plot(idx_tampil, hasil_cnn['y_pred'][idx_tampil], marker='o', color='blue', label='CNN')
plt.plot(idx_tampil, hasil_lstm['y_pred'][idx_tampil], marker='o', color='green', label='LSTM')
plt.yticks([0, 1], label_names_short)
plt.xlabel('Data ke-')
plt.ylabel('Label')
plt.title('Plot Hasil Testing Suka vs Tidak Suka')
plt.legend()
plt.tight_layout()
plt.show()

# 6.3 Ringkasan Perbandingan CNN vs LSTM
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
print(f"\n>> Model terbaik berdasarkan F1-Score untuk klasifikasi Suka vs Tidak Suka")
print(f"   adalah: {best_model_name}")

# 7. Output
print("\n" + "="*60)
print("7. OUTPUT")
print("="*60)

model_terbaik = model_cnn if best_model_name == 'CNN' else model_lstm

buku_lookup = books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']].drop_duplicates('ISBN')

print("menggunakan model_terbaik, age_scaler, year_scaler, author_stats,")
print("publisher_stats, isbn_stats, global_mean_train, fitur_kolom, buku_lookup.")


def cari_isbn_dari_judul(judul_buku):
    # Mencari ISBN dari judul buku (pencarian tidak case-sensitive & partial)
    hasil = buku_lookup[buku_lookup['Book-Title'].str.contains(judul_buku, case=False, na=False)]
    if len(hasil) == 0:
        return None
    return hasil.iloc[0]['ISBN']


def prediksi_suka_buku(judul_buku, nama_author, tahun_terbit, nama_publisher, usia,
                        model=None):
    #Memprediksi apakah seorang pembaca akan menyukai sebuah buku,
    #berdasarkan input: judul buku, author, tahun terbit, publisher, usia.
    if model is None:
        model = model_terbaik

    isbn_ditemukan = cari_isbn_dari_judul(judul_buku)

    if isbn_ditemukan is not None and isbn_ditemukan in isbn_stats.index:
        isbn_target_enc = isbn_stats.loc[isbn_ditemukan, 'ISBN_TargetEnc']
        isbn_popularitas = isbn_stats.loc[isbn_ditemukan, 'ISBN_LogCount_norm']
        buku_ditemukan = True
        pesan_buku = f"Buku ditemukan di data historis (ISBN: {isbn_ditemukan})."
        print(f"[Info] {pesan_buku}")
    else:
        isbn_target_enc = global_mean_train
        isbn_popularitas = 0.0
        buku_ditemukan = False
        pesan_buku = "Buku tidak ditemukan"
        print(f"[Info] {pesan_buku} (buku baru/cold-start, memakai nilai default rata-rata global).")

    author_target_enc = author_stats['Author_TargetEnc'].get(nama_author, global_mean_train)
    publisher_target_enc = publisher_stats['Publisher_TargetEnc'].get(nama_publisher, global_mean_train)

    age_norm = age_scaler.transform(pd.DataFrame([[usia]], columns=['Age']))[0][0]
    year_norm = year_scaler.transform(
        pd.DataFrame([[tahun_terbit]], columns=['Year-Of-Publication'])
    )[0][0]

    fitur_dict = {
        'Age_normalized': age_norm,
        'Author_TargetEnc': author_target_enc,
        'Publisher_TargetEnc': publisher_target_enc,
        'Year_normalized': year_norm,
        'prev': 0.5,                    # netral, belum ada riwayat rating
        'dist': global_mean_train,      # netral, belum ada riwayat rating
        'ISBN_TargetEnc': isbn_target_enc,
        'ISBN_Popularitas': isbn_popularitas
    }

    vektor = np.array([fitur_dict[k] for k in fitur_kolom]).reshape(1, len(fitur_kolom), 1)
    probabilitas = float(model.predict(vektor, verbose=0)[0][0])
    kelas = 'Suka' if probabilitas >= 0.5 else 'Tidak Suka'

    return {
        'kelas': kelas,
        'probabilitas_suka': probabilitas,
        'buku_ditemukan': buku_ditemukan,
        'pesan_buku': pesan_buku
    }

# 7.1 Input dari Pengguna
print("\n" + "-"*50)
print("INPUT DATA UNTUK PREDIKSI")
print("-"*50)


def minta_input_angka(pertanyaan):
    # minta input angka dari user kalau formatnya salah
    while True:
        nilai_teks = input(pertanyaan).strip()
        try:
            return float(nilai_teks)
        except ValueError:
            print(f"Input '{nilai_teks}' bukan angka yang valid. Coba lagi (contoh: 2001, 25).")


def minta_input_teks(pertanyaan):
    # Minta input teks dari user kalau kosong
    while True:
        nilai_teks = input(pertanyaan).strip()
        if nilai_teks != "":
            return nilai_teks
        print("Input tidak boleh kosong. Coba lagi.")


judul_input = minta_input_teks("Judul buku: ")
author_input = minta_input_teks("Nama penulis (author): ")
tahun_input = minta_input_angka("Tahun terbit buku: ")
publisher_input = minta_input_teks("Nama penerbit (publisher): ")
usia_input = minta_input_angka("Usia pembaca: ")

hasil_prediksi = prediksi_suka_buku(judul_input, author_input, tahun_input,
                                     publisher_input, usia_input)
print(f"\nHasil Prediksi: {hasil_prediksi}")

# 7.2 REST API Dihubungkan ke Mobile
flask_api_code = '''
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model_terbaik.h5')
age_scaler = joblib.load('age_scaler.pkl')
year_scaler = joblib.load('year_scaler.pkl')
author_lookup = joblib.load('author_lookup.pkl')
publisher_lookup = joblib.load('publisher_lookup.pkl')
isbn_lookup = joblib.load('isbn_lookup.pkl')
global_mean_train = joblib.load('global_mean_train.pkl')
fitur_kolom = joblib.load('fitur_kolom.pkl')
buku_lookup = joblib.load('buku_lookup.pkl')

FIELD_WAJIB = ['judul_buku', 'author', 'tahun_terbit', 'publisher', 'usia']


def cari_isbn_dari_judul(judul_buku):
    hasil = buku_lookup[buku_lookup['Book-Title'].str.contains(judul_buku, case=False, na=False)]
    if len(hasil) == 0:
        return None
    return hasil.iloc[0]['ISBN']


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            'error': 'Body request tidak valid. Kirim JSON, contoh: '
                     '{"judul_buku": "...", "author": "...", "tahun_terbit": 2001, '
                     '"publisher": "...", "usia": 25}'
        }), 400

    field_hilang = [f for f in FIELD_WAJIB if f not in data or str(data[f]).strip() == '']
    if field_hilang:
        return jsonify({
            'error': f"Field berikut wajib diisi dan tidak boleh kosong: {field_hilang}"
        }), 400

    judul_buku = str(data['judul_buku']).strip()
    nama_author = str(data['author']).strip()
    nama_publisher = str(data['publisher']).strip()

    try:
        tahun_terbit = float(data['tahun_terbit'])
    except (ValueError, TypeError):
        return jsonify({
            'error': f"tahun_terbit harus berupa angka, diterima: {data['tahun_terbit']!r}"
        }), 400

    try:
        usia = float(data['usia'])
    except (ValueError, TypeError):
        return jsonify({
            'error': f"usia harus berupa angka, diterima: {data['usia']!r}"
        }), 400

    isbn_ditemukan = cari_isbn_dari_judul(judul_buku)

    if isbn_ditemukan is not None and isbn_ditemukan in isbn_lookup.index:
        isbn_target_enc = isbn_lookup.loc[isbn_ditemukan, 'ISBN_TargetEnc']
        isbn_popularitas = isbn_lookup.loc[isbn_ditemukan, 'ISBN_LogCount_norm']
        buku_ditemukan = True
        pesan_buku = f"Buku ditemukan di data historis (ISBN: {isbn_ditemukan})."
    else:
        isbn_target_enc = global_mean_train
        isbn_popularitas = 0.0
        buku_ditemukan = False
        pesan_buku = "Buku tidak ditemukan"

    author_target_enc = author_lookup.get(nama_author, global_mean_train)
    publisher_target_enc = publisher_lookup.get(nama_publisher, global_mean_train)

    age_norm = age_scaler.transform(pd.DataFrame([[usia]], columns=['Age']))[0][0]
    year_norm = year_scaler.transform(
        pd.DataFrame([[tahun_terbit]], columns=['Year-Of-Publication'])
    )[0][0]

    fitur_dict = {
        'Age_normalized': age_norm,
        'Author_TargetEnc': author_target_enc,
        'Publisher_TargetEnc': publisher_target_enc,
        'Year_normalized': year_norm,
        'prev': 0.5,
        'dist': global_mean_train,
        'ISBN_TargetEnc': isbn_target_enc,
        'ISBN_Popularitas': isbn_popularitas
    }

    vektor = np.array([fitur_dict[k] for k in fitur_kolom]).reshape(1, len(fitur_kolom), 1)
    probabilitas = float(model.predict(vektor, verbose=0)[0][0])
    kelas = 'Suka' if probabilitas >= 0.5 else 'Tidak Suka'

    return jsonify({
        'kelas': kelas,
        'probabilitas_suka': probabilitas,
        'buku_ditemukan': buku_ditemukan,
        'pesan_buku': pesan_buku
    })


@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({'error': 'Terjadi kesalahan pada server saat memproses prediksi.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''
