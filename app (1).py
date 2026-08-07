import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import tempfile
import os

# ==========================================
# 1. ARSITEKTUR MODEL (Disamakan dengan Training!)
# ==========================================
class SimplePointNet(nn.Module):
    def __init__(self, num_classes):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] 
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ==========================================
# 2. TAMPILAN WEB & PROSES PREDIKSI
# ==========================================
st.title("🚀 Inspektur Wadah 3D AI")
st.write("Upload file .pcd kamu di sini untuk dianalisis otomatis!")

password = st.sidebar.text_input("Masukkan Password", type="password")
if password != "wadahaman123":
    st.warning("Silakan masukkan password di menu sebelah kiri untuk mengakses AI.")
    st.stop()

uploaded_file = st.file_uploader("Pilih file .pcd", type=['pcd'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.info("AI sedang memproses file 3D...")

    try:
        device = torch.device("cpu")
        checkpoint = torch.load("best_model.pth", map_location=device)
        
        # LOGIKA ANTI-ERROR JIKA BENTUK FILE PTH BERBEDA
        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            classes = checkpoint["classes"]
            model_state = checkpoint["model_state"]
        else:
            # GANTI DAFTAR INI KALAU NAMA FOLDER KELASMU BEDA YA!
            classes = ["Cacat_Dinding", "Cacat_Lantai", "Valid"] 
            model_state = checkpoint
            
        model = SimplePointNet(len(classes)).to(device)
        model.load_state_dict(model_state)
        model.eval()

        # PREPROCESSING (Sama persis dengan LidarDataset)
        pcd = o3d.io.read_point_cloud(tmp_path)
        points = np.asarray(pcd.points)

        max_points = 1024
        num_points = points.shape[0]
        if num_points == 0:
            points = np.zeros((max_points, 3))
        elif num_points >= max_points:
            indices = np.random.choice(num_points, max_points, replace=False)
            points = points[indices]
        else:
            indices = np.random.choice(num_points, max_points, replace=True)
            points = points[indices]

        # Normalisasi ke tengah (Titik 0,0,0)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if m > 0:
            points = points / m

        points_tensor = torch.tensor(points, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

        # PROSES PREDIKSI
        with torch.no_grad():
            prediksi = model(points_tensor)
            persentase = F.softmax(prediksi, dim=1).squeeze() * 100

        st.subheader("Hasil Analisis:")
        skor_per_kelas = [(classes[i], persentase[i].item()) for i in range(len(classes))]
        skor_per_kelas.sort(key=lambda x: x[1], reverse=True)

        top_1_kelas = skor_per_kelas[0][0]

        if top_1_kelas.lower() == "valid":
            st.success(f"✅ Barang ini Valid Sempurna! (Keyakinan: {skor_per_kelas[0][1]:.2f}%)")
        else:
            st.error("⚠️ Peringatan: Barang terdeteksi CACAT dengan alasan berikut:")
            for nama_kelas, skor in skor_per_kelas:
                # Tampilkan semua dugaan cacat yang peluangnya di atas 15%
                if nama_kelas.lower() != "valid" and skor >= 15.0:
                    st.write(f"- **{nama_kelas}** (Confidence: {skor:.2f}%)")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
