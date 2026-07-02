# ============================================
# LEGO COLOR CHALLENGE – 14 FEATURES + CNN 11x11 AUTOENCODER
# Klasyczne modele + Autoencoder (MLP) + CNN Autoencoder + Transformer + GUI
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from PIL import Image

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ============================================
# 0. KONFIGURACJA
# ============================================

DATA_PATH = "archive"
FILE_MAIN = os.path.join(DATA_PATH, "legocolor-extended.csv")
FILE_COLORS = os.path.join(DATA_PATH, "colors.csv")

DEVICE = torch.device("cpu")
print(f">>> Używane urządzenie: {DEVICE}")

# ============================================
# 0A. DEFINICJA 14 CECH INFORMACYJNYCH
# ============================================

FEATURE_COLS_14 = [
    "R", "G", "B",                  # 3
    "Hue", "Saturation", "Value",  # 6
    "Y", "U", "V",                 # 9
    "R_norm", "G_norm", "B_norm",  # 12
    "Dist_RG", "SatIndex"          # 14
]


def compute_color_spaces(r, g, b):
    # HSV
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h_deg = h * 360.0
    s_255 = s * 255.0
    v_255 = v * 255.0

    # YUV
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v_yuv = 0.615 * r - 0.51499 * g - 0.10001 * b

    # normalized RGB
    s_rgb = r + g + b
    if s_rgb == 0:
        r_norm = g_norm = b_norm = 0.0
    else:
        r_norm = r / s_rgb
        g_norm = g / s_rgb
        b_norm = b / s_rgb

    # dystans między kanałami (przykładowo RG)
    dist_rg = abs(r - g)

    # indeks nasycenia (prosty, od 0 do 1)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    sat_index = 0.0 if max_c == 0 else (max_c - min_c) / max_c

    return {
        "R": r,
        "G": g,
        "B": b,
        "Hue": h_deg,
        "Saturation": s_255,
        "Value": v_255,
        "Y": y,
        "U": u,
        "V": v_yuv,
        "R_norm": r_norm,
        "G_norm": g_norm,
        "B_norm": b_norm,
        "Dist_RG": dist_rg,
        "SatIndex": sat_index
    }


def get_pixel_features_14(r, g, b, feature_cols=FEATURE_COLS_14):
    """Zwraca 14-cechowy wektor dla pojedynczego piksela."""
    fd = compute_color_spaces(r, g, b)
    return np.array([[fd[col] for col in feature_cols]], dtype=np.float32)


# ============================================
# 0B. CNN AUTOENCODER NA PATCHACH 11x11 (Z TEKSTURĄ I WYCINKAMI)
# ============================================

PATCH_SIZE = 11
LATENT_DIM_CNN = 8


def extract_real_patch(img_np, x, y, patch_size=11):
    h, w, _ = img_np.shape
    half = patch_size // 2

    y_start = max(0, y - half)
    y_end = min(h, y + half + 1)
    x_start = max(0, x - half)
    x_end = min(w, x + half + 1)

    patch = img_np[y_start:y_end, x_start:x_end]

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        padded_patch = np.zeros((patch_size, patch_size, 3), dtype=img_np.dtype)
        t_y_start = half - (y - y_start)
        t_y_end = t_y_start + (y_end - y_start)
        t_x_start = half - (x - x_start)
        t_x_end = t_x_start + (x_end - x_start)
        padded_patch[t_y_start:t_y_end, t_x_start:t_x_end] = patch
        patch = padded_patch

    return patch.transpose(2, 0, 1).astype(np.float32) / 255.0


def generate_textured_patch(r, g, b, patch_size=11):
    patch = np.zeros((3, patch_size, patch_size), dtype=np.float32)
    patch[0, :, :] = r / 255.0
    patch[1, :, :] = g / 255.0
    patch[2, :, :] = b / 255.0

    # Generowanie delikatnego gradientu imitującego załamanie światła na plastiku
    x = np.linspace(-1, 1, patch_size)
    y = np.linspace(-1, 1, patch_size)
    xv, yv = np.meshgrid(x, y)
    gradient = (xv + yv) * 0.02
    
    for c in range(3):
        patch[c, :, :] += gradient

    # Sztuczny szum cyfrowy matrycy aparatu
    noise = np.random.normal(0, 0.015, (3, patch_size, patch_size))
    patch += noise
    
    return np.clip(patch, 0.0, 1.0)


class PatchDataset(Dataset):
    def __init__(self, X_patches):
        self.X_patches = X_patches

    def __len__(self):
        return self.X_patches.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X_patches[idx], dtype=torch.float32)


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM_CNN):
        super().__init__()
        # Encoder: 11x11 -> 5x5 -> 2x2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 11x11 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 11x11 -> 5x5
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 5x5 -> 5x5
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 5x5 -> 2x2
        )
        self.enc_fc = nn.Linear(32 * 2 * 2, latent_dim)

        # Decoder: 2x2 -> 5x5 -> 11x11
        self.dec_fc = nn.Linear(latent_dim, 32 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 3,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.enc_fc(z)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(x.size(0), 32, 2, 2)
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def train_cnn_autoencoder(X_patches, epochs=20, batch_size=128, lr=1e-3):
    print("\n>>> Trening CNN autoenkodera na patchach przestrzennych...")
    dataset = PatchDataset(X_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNAutoencoder(latent_dim=LATENT_DIM_CNN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            x_rec, _ = model(batch)
            loss = criterion(x_rec, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{epochs} | CNN AE loss: {avg_loss:.6f}")

    return model


def encode_with_cnn_autoencoder(model, X_patches, batch_size=256):
    dataset = PatchDataset(X_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            z = model.encode(batch)
            zs.append(z.cpu().numpy())
    Z = np.vstack(zs)
    return Z


# ============================================
# 1. GUI – WYBÓR ZDJĘCIA + KLIKANIE PIKSELI
# ============================================

def gui_image_tester(model_14, scaler_14, label_encoder,
                     feature_cols_14,
                     cnn_model=None, rf_cnn=None, scaler_cnn=None):
    def choose_file():
        file_path = filedialog.askopenfilename(
            title="Wybierz zdjęcie LEGO",
            filetypes=[("Obrazy", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")
        img.thumbnail((900, 900))
        img_np = np.array(img)

        tk_img = ImageTk.PhotoImage(img)
        canvas.img = tk_img
        canvas.img_np = img_np
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.config(width=img.width, height=img.height)

        info_label.config(text="Kliknij w piksel, aby przewidzieć kolor LEGO")

    def on_click(event):
        if not hasattr(canvas, "img_np"):
            return

        x, y = event.x, event.y
        img_np = canvas.img_np

        if y >= img_np.shape[0] or x >= img_np.shape[1]:
            return

        r, g, b = map(int, img_np[y, x])

        # 14-cechowy wektor
        X_sample_14 = get_pixel_features_14(r, g, b, feature_cols_14)
        X_scaled_14 = scaler_14.transform(X_sample_14)
        y_pred_14 = model_14.predict(X_scaled_14)
        color_14 = label_encoder.inverse_transform(y_pred_14)[0]

        text = f"RGB=({r},{g},{b}) → 14F: {color_14}"

        # Predykcja z CNN latent (Wycięcie prawdziwego kontekstu klocka)
        if cnn_model is not None and rf_cnn is not None and scaler_cnn is not None:
            real_patch = extract_real_patch(img_np, x, y, patch_size=PATCH_SIZE)
            patch_t = torch.tensor(real_patch, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z = cnn_model.encode(patch_t).cpu().numpy()
            z_scaled = scaler_cnn.transform(z)
            y_pred_cnn = rf_cnn.predict(z_scaled)
            color_cnn = label_encoder.inverse_transform(y_pred_cnn)[0]
            text += f" | CNN: {color_cnn}"

        result_label.config(text=text, fg="blue")
        canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", outline="red")

    root = tk.Tk()
    root.title("LEGO Color Detector – GUI (14F + CNN)")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    btn = tk.Button(frame, text="Wybierz zdjęcie", command=choose_file, font=("Arial", 12))
    btn.pack()

    info_label = tk.Label(root, text="Wybierz zdjęcie, aby rozpocząć", font=("Arial", 11))
    info_label.pack(pady=5)

    canvas = tk.Canvas(root, bg="gray")
    canvas.pack()

    canvas.bind("<Button-1>", on_click)

    result_label = tk.Label(root, text="", font=("Arial", 12))
    result_label.pack(pady=10)

    root.mainloop()


# ============================================
# 2. WCZYTANIE DANYCH
# ============================================

def load_data():
    print(">>> Wczytywanie danych...")
    df_data = pd.read_csv(FILE_MAIN, delimiter=";")
    print("Kształt df_data:", df_data.shape)

    if os.path.exists(FILE_COLORS):
        df_color = pd.read_csv(FILE_COLORS, delimiter=",")
        print("Kształt df_color:", df_color.shape)
    else:
        df_color = None
        print("UWAGA: colors.csv nie znaleziony – pomijam.")

    return df_data, df_color


# ============================================
# 3. PRZYGOTOWANIE DANYCH (14 CECH)
# ============================================

def prepare_data(df_data, df_color):
    print("\n>>> Przygotowanie danych...")

    if "Color" not in df_data.columns:
        raise ValueError("Brak kolumny 'Color' w legocolor-extended.csv")

    for c in ["R", "G", "B"]:
        df_data[c] = pd.to_numeric(df_data[c], errors="coerce")

    df_data = df_data.dropna(subset=["R", "G", "B", "Color"])
    df_data[["R", "G", "B"]] = df_data[["R", "G", "B"]].astype(int)

    print("Generowanie 14 cech informacyjnych...")
    feats = {col: [] for col in FEATURE_COLS_14}
    for r, g, b in zip(df_data["R"], df_data["G"], df_data["B"]):
        fd = compute_color_spaces(r, g, b)
        for col in FEATURE_COLS_14:
            feats[col].append(fd[col])

    for col in FEATURE_COLS_14:
        df_data[col] = feats[col]

    print("Dostępne klasy (kolory):", df_data["Color"].unique())
    print("Używane kolumny cech:", FEATURE_COLS_14)

    return df_data


# ============================================
# 4. PODZIAŁ I ENKODOWANIE
# ============================================

def split_and_encode(df_data, test_size=0.2, random_state=42):
    print("\n>>> Podział na zbiór treningowy i testowy...")

    y = df_data["Color"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    indices = np.arange(len(df_data))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    X_train = df_data.iloc[train_idx][FEATURE_COLS_14].values
    X_test = df_data.iloc[test_idx][FEATURE_COLS_14].values

    print("Rozmiar X_train:", X_train.shape)
    print("Rozmiar X_test :", X_test.shape)
    print("Liczba klas:", len(le.classes_))

    return X_train, X_test, y_train, y_test, le, train_idx, test_idx


# ============================================
# 5. DATASETY TORCH (TABLICOWE)
# ============================================

class FeatureDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ============================================
# 6. AUTOENCODER (MLP) NA 14 CECHACH
# ============================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


def train_autoencoder(X_train_scaled, input_dim, latent_dim=8, epochs=20, batch_size=128, lr=1e-3):
    print("\n>>> Trening MLP autoenkodera na 14 cechach...")
    dataset = FeatureDataset(X_train_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            x_rec, _ = model(batch)
            loss = criterion(x_rec, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{epochs} | AE loss: {avg_loss:.6f}")

    return model


def encode_with_autoencoder(model, X_scaled):
    model.eval()
    dataset = FeatureDataset(X_scaled)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    zs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            _, z = model(batch)
            zs.append(z.cpu().numpy())
    Z = np.vstack(zs)
    return Z


# ============================================
# 7. TRANSFORMEROWY KLASYFIKATOR CECH
# ============================================

class FeatureTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.cls_head(x)
        return logits


def train_transformer_classifier(X_train_scaled, y_train, input_dim, num_classes,
                                 epochs=20, batch_size=128, lr=1e-3):
    print("\n>>> Trening transformerowego klasyfikatora cech...")
    dataset = FeatureDataset(X_train_scaled, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FeatureTransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        avg_loss = total_loss / len(dataset)
        acc = correct / total
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    return model


def evaluate_transformer(model, X_test_scaled, y_test):
    model.eval()
    dataset = FeatureDataset(X_test_scaled, y_test)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n>>> Transformer – wyniki na zbiorze testowym:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return acc, prec, rec, f1, y_pred


# ============================================
# 8. TRENING KLASYCZNYCH MODELI (14 CECH)
# ============================================

def train_and_select_classical_models(X_train, y_train):
    print("\n>>> Trening klasycznych modeli i wybór najlepszego...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # USUNIĘTO multi_class="auto" (zapobieganie FutureWarning w scikit-learn >= 1.5)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC_RBF": SVC(kernel="rbf", C=5, gamma="scale", probability=False),
        "LinearSVC": LinearSVC(C=1.0),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "GaussianNB": GaussianNB()
    }

    results = []

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
            mean_acc = scores.mean()
            results.append((name, mean_acc))
            print(f"Model: {name:18s} | CV Accuracy: {mean_acc:.4f}")
        except Exception as e:
            print(f"Model: {name:18s} | BŁĄD w treningu: {e}")

    results = [r for r in results if not np.isnan(r[1])]
    results.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = results[0]
    best_model = models[best_name]
    best_model.fit(X_train_scaled, y_train)

    print(f"\n>>> Najlepszy klasyczny model: {best_name} (CV Accuracy = {best_score:.4f})")

    return best_model, scaler, results, X_train_scaled


# ============================================
# 9. EWALUACJA KLASYCZNEGO MODELU
# ============================================

def evaluate_model(model, scaler, X_test, y_test, label_encoder):
    print("\n>>> Ewaluacja najlepszego klasycznego modelu...")

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n>>> Raport klasyfikacji (klasyczny model):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix – klasyczny model (14 cech)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return acc, prec, rec, f1, y_pred


# ============================================
# 10. FEATURE IMPORTANCE
# ============================================

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(6, 4))
        sns.barplot(x=importances, y=feature_names, orient="h")
        plt.title("Feature Importance (model drzewiasty)")
        plt.tight_layout()
        plt.show()
    else:
        print("Model nie posiada atrybutu feature_importances_ – pomijam wykres.")


# ============================================
# 11. WYKRES 3D RGB
# ============================================

def plot_3d_rgb(df_data):
    if not {"R", "G", "B", "Color"}.issubset(df_data.columns):
        print("Brak R,G,B lub Color – pomijam wykres 3D.")
        return

    print("\n>>> Wykres 3D w przestrzeni RGB...")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    classes = df_data["Color"].unique()
    colors = plt.colormaps.get_cmap("tab20")

    for i, cls in enumerate(classes):
        subset = df_data[df_data["Color"] == cls]
        ax.scatter(subset["R"], subset["G"], subset["B"],
                   color=colors(i / len(classes)), label=cls, s=10)

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("LEGO colors in RGB space")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================
# 12. TEST NA ZDJĘCIU (MATPLOTLIB)
# ============================================

def open_image_and_click(model_14, scaler_14, label_encoder, image_path,
                         feature_cols_14,
                         cnn_model=None, rf_cnn=None, scaler_cnn=None):
    print(f"\n>>> Ładowanie obrazu: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_title("Kliknij w piksel z kolorem klocka")
    ax.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)
        r, g, b = map(int, img_np[y, x])
        print(f"\nKliknięto w punkt: (x={x}, y={y}) | RGB=({r},{g},{b})")

        X_sample_14 = get_pixel_features_14(r, g, b, feature_cols_14)
        X_sample_scaled_14 = scaler_14.transform(X_sample_14)
        y_pred_14 = model_14.predict(X_sample_scaled_14)
        color_14 = label_encoder.inverse_transform(y_pred_14)[0]
        print(f"Przewidywany kolor (14F): {color_14}")

        # Poprawne wycinanie rzeczywistego kontekstu przestrzennego
        if cnn_model is not None and rf_cnn is not None and scaler_cnn is not None:
            real_patch = extract_real_patch(img_np, x, y, patch_size=PATCH_SIZE)
            patch_t = torch.tensor(real_patch, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z = cnn_model.encode(patch_t).cpu().numpy()
            z_scaled = scaler_cnn.transform(z)
            y_pred_cnn = rf_cnn.predict(z_scaled)
            color_cnn = label_encoder.inverse_transform(y_pred_cnn)[0]
            print(f"Przewidywany kolor (CNN latent): {color_cnn}")

        ax.plot(x, y, "ro", markersize=5)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


# ============================================
# 13. RAPORT TEKSTOWY
# ============================================

def generate_text_report(best_classical_name, cv_results_classical,
                         metrics_classical, metrics_ae, metrics_tr,
                         metrics_cnn_rf):
    acc_c, prec_c, rec_c, f1_c = metrics_classical
    acc_ae, prec_ae, rec_ae, f1_ae = metrics_ae
    acc_tr, prec_tr, rec_tr, f1_tr = metrics_tr
    acc_cnn, prec_cnn, rec_cnn, f1_cnn = metrics_cnn_rf

    print("\n" + "=" * 70)
    print("RAPORT PODSUMOWUJĄCY MODELE")
    print("=" * 70)

    print("\nModele klasyczne (CV Accuracy) – 14 cech:")
    for name, score in cv_results_classical:
        print(f" - {name:18s}: {score:.4f}")

    print(f"\nNajlepszy klasyczny model: {best_classical_name}")
    print("Wyniki na zbiorze testowym (klasyczny, 14F):")
    print(f" - Accuracy : {acc_c:.4f}")
    print(f" - Precision: {prec_c:.4f}")
    print(f" - Recall   : {rec_c:.4f}")
    print(f" - F1-score : {f1_c:.4f}")

    print("\nAutoencoder (MLP) + RF – latent z 14 cech:")
    print(f" - Accuracy : {acc_ae:.4f}")
    print(f" - Precision: {prec_ae:.4f}")
    print(f" - Recall   : {rec_ae:.4f}")
    print(f" - F1-score : {f_ae:.4f}" if (f_ae := f1_ae) or True else "")

    print("\nTransformerowy klasyfikator cech (14F):")
    print(f" - Accuracy : {acc_tr:.4f}")
    print(f" - Precision: {prec_tr:.4f}")
    print(f" - Recall   : {rec_tr:.4f}")
    print(f" - F1-score : {f1_tr:.4f}")

    print("\nCNN Autoencoder 11x11 + RF – latent z patchy:")
    print(f" - Accuracy : {acc_cnn:.4f}")
    print(f" - Precision: {prec_cnn:.4f}")
    print(f" - Recall   : {rec_cnn:.4f}")
    print(f" - F1-score : {f1_cnn:.4f}")

    print("\nInterpretacja:")
    print(" - 14 cech informacyjnych dają stabilną bazę.")
    print(" - MLP autoencoder kompresuje 14F do latentnej reprezentacji, na której działa RF.")
    print(" - Transformer traktuje wektor cech jak sekwencję i wykorzystuje mechanizm uwagi.")
    print(" - CNN autoencoder uczy się nieliniowej reprezentacji z uwzględnieniem tekstury i szumu przestrzennego.")
    print("=" * 70 + "\n")


# ============================================
# 14. GŁÓWNY PRZEPŁYW
# ============================================

def main():
    df_data, df_color = load_data()
    df_data = prepare_data(df_data, df_color)

    plot_3d_rgb(df_data)

    X_train, X_test, y_train, y_test, le, train_idx, test_idx = split_and_encode(df_data)

    # 1) Klasyczne modele na 14 cechach
    best_model, scaler_14, cv_results_classical, X_train_scaled_14 = train_and_select_classical_models(X_train, y_train)

    acc_c, prec_c, rec_c, f1_c, y_pred_classical = evaluate_model(
        best_model, scaler_14, X_test, y_test, le
    )

    plot_feature_importance(best_model, FEATURE_COLS_14)

    # 2) MLP autoencoder na 14 cechach
    input_dim = X_train_scaled_14.shape[1]
    ae_model = train_autoencoder(X_train_scaled_14, input_dim=input_dim, latent_dim=8, epochs=20)

    X_train_ae = encode_with_autoencoder(ae_model, X_train_scaled_14)
    X_test_scaled_14 = scaler_14.transform(X_test)
    X_test_ae = encode_with_autoencoder(ae_model, X_test_scaled_14)

    rf_ae = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_ae.fit(X_train_ae, y_train)
    y_pred_ae = rf_ae.predict(X_test_ae)

    acc_ae = accuracy_score(y_test, y_pred_ae)
    prec_ae = precision_score(y_test, y_pred_ae, average="weighted", zero_division=0)
    rec_ae = recall_score(y_test, y_pred_ae, average="weighted", zero_division=0)
    f1_ae = f1_score(y_test, y_pred_ae, average="weighted", zero_division=0)

    print("\n>>> Autoencoder (MLP) + RF – wyniki na zbiorze testowym:")
    print(f"Accuracy : {acc_ae:.4f}")
    print(f"Precision: {prec_ae:.4f}")
    print(f"Recall   : {rec_ae:.4f}")
    print(f"F1-score : {f1_ae:.4f}")

    # 3) Transformer na 14 cechach
    tr_model = train_transformer_classifier(
        X_train_scaled_14, y_train,
        input_dim=input_dim,
        num_classes=len(le.classes_),
        epochs=15
    )

    acc_tr, prec_tr, rec_tr, f1_tr, y_pred_tr = evaluate_transformer(
        tr_model, X_test_scaled_14, y_test
    )

    # 4) POPRAWIONE: Generowanie zróżnicowanych strukturalnie patchy do CNN
    print("\n>>> Przygotowanie strukturalnych patchy z teksturą do CNN...")
    X_train_rgb = df_data.iloc[train_idx][["R", "G", "B"]].values
    X_test_rgb = df_data.iloc[test_idx][["R", "G", "B"]].values

    # Generowanie patchy bogatych w gradient i mikroszum plastikowej powierzchni klocka
    X_train_patches = np.array([generate_textured_patch(r, g, b, PATCH_SIZE) for r, g, b in X_train_rgb], dtype=np.float32)
    X_test_patches = np.array([generate_textured_patch(r, g, b, PATCH_SIZE) for r, g, b in X_test_rgb], dtype=np.float32)

    cnn_ae = train_cnn_autoencoder(X_train_patches, epochs=15)

    Z_train_cnn = encode_with_cnn_autoencoder(cnn_ae, X_train_patches)
    Z_test_cnn = encode_with_cnn_autoencoder(cnn_ae, X_test_patches)

    scaler_cnn = StandardScaler()
    Z_train_cnn_scaled = scaler_cnn.fit_transform(Z_train_cnn)
    Z_test_cnn_scaled = scaler_cnn.transform(Z_test_cnn)

    rf_cnn = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_cnn.fit(Z_train_cnn_scaled, y_train)
    y_pred_cnn = rf_cnn.predict(Z_test_cnn_scaled)

    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    prec_cnn = precision_score(y_test, y_pred_cnn, average="weighted", zero_division=0)
    rec_cnn = recall_score(y_test, y_pred_cnn, average="weighted", zero_division=0)
    f1_cnn = f1_score(y_test, y_pred_cnn, average="weighted", zero_division=0)

    print("\n>>> CNN AE + RF – wyniki na zbiorze testowym:")
    print(f"Accuracy : {acc_cnn:.4f}")
    print(f"Precision: {prec_cnn:.4f}")
    print(f"Recall   : {rec_cnn:.4f}")
    print(f"F1-score : {f1_cnn:.4f}")

    best_classical_name = type(best_model).__name__
    generate_text_report(
        best_classical_name,
        cv_results_classical,
        (acc_c, prec_c, rec_c, f1_c),
        (acc_ae, prec_ae, rec_ae, f1_ae),
        (acc_tr, prec_tr, rec_tr, f1_tr),
        (acc_cnn, prec_cnn, rec_cnn, f1_cnn)
    )

    image_path = "lego_photo.jpg"
    if os.path.exists(image_path):
        open_image_and_click(
            best_model, scaler_14, le, image_path,
            FEATURE_COLS_14,
            cnn_model=cnn_ae, rf_cnn=rf_cnn, scaler_cnn=scaler_cnn
        )
    else:
        print(f"\nUWAGA: Plik {image_path} nie istnieje – pomijam test na zdjęciu.")

    return best_model, scaler_14, le, FEATURE_COLS_14, cnn_ae, rf_cnn, scaler_cnn


if __name__ == "__main__":
    best_model, scaler_14, le, feature_cols_14, cnn_ae, rf_cnn, scaler_cnn = main()
    print("\n>>> Uruchamiam GUI do testowania zdjęć...")
    gui_image_tester(best_model, scaler_14, le, feature_cols_14, cnn_model=cnn_ae, rf_cnn=rf_cnn, scaler_cnn=scaler_cnn)