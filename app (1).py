import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import tempfile
import os

# ==========================================
# 1. FUNGSI PREPROCESSING (FPS 2048)
# ==========================================
def farthest_point_sample(points, npoint):
    N, _ = points.shape
    centroids = np.zeros(npoint, dtype=np.int32)
    distances = np.ones(N) * 1e10
    farthest = 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return points[centroids]

# ==========================================
# 2. ARSITEKTUR MODEL DGCNN (Monster)
# ==========================================
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN_MultiLabel(nn.Module):
    def __init__(self, num_classes, k=20):
        super(DGCNN_MultiLabel, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

# ==========================================
# 3. TAMPILAN WEB & PROSES PREDIKSI
# ==========================================
st.set_page_config(page_title="Scanner Wadah 3D", page_icon="🛻")
st.title("Validasi 🛻")
st.write("Upload file mentah (.pcd)")

password = st.sidebar.text_input("Masukkan Password", type="password")
if password != "abc123":
    st.warning("Silakan masukkan password di menu sebelah kiri")
    st.stop()

uploaded_file = st.file_uploader("Pilih file .pcd", type=['pcd'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.info("🔄 sedang memproses dan memindai file 3D... (Harap tunggu beberapa detik)")

    try:
        device = torch.device("cpu")
        
        # 🚨 PASTIKAN NAMA FILE PTH INI SAMA DENGAN HASIL TRAINING TERAKHIRMU!
        checkpoint = torch.load("best_model_multilabell.pth", map_location=device)
        
        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            classes = checkpoint["classes"]
            model_state = checkpoint["model_state"]
        else:
            st.error("Model PTH tidak valid atau tidak memiliki daftar kelas!")
            st.stop()
            
        model = DGCNN_MultiLabel(len(classes)).to(device)
        model.load_state_dict(model_state)
        model.eval()

        # ==========================================
        # 4. PREPROCESSING OTOMATIS
        # ==========================================
        pcd = o3d.io.read_point_cloud(tmp_path)
        points = np.asarray(pcd.points)

        TARGET_POINTS = 2048
        num_points = points.shape[0]
        
        if num_points == 0:
            points = np.zeros((TARGET_POINTS, 3))
        elif num_points >= TARGET_POINTS:
            # Gunakan FPS persis seperti offline
            points = farthest_point_sample(points, TARGET_POINTS)
        else:
            indices = np.random.choice(num_points, TARGET_POINTS, replace=True)
            points = points[indices]

        # Normalisasi ke tengah (Titik 0,0,0)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if m > 0:
            points = points / m

        points_tensor = torch.tensor(points, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

        # ==========================================
        # 5. PREDIKSI MULTI-LABEL
        # ==========================================
        # ==========================================
        # 5. PREDIKSI MULTI-LABEL SESUAI ATURAN BARU
        # ==========================================
        with torch.no_grad():
            prediksi = model(points_tensor)
            probs = torch.sigmoid(prediksi).squeeze() * 100

        st.subheader("📊 Hasil Analisis AI:")
        
        # Urutkan semua skor per kelas dari terbesar ke terkecil
        skor_per_kelas = [(classes[i], probs[i].item()) for i in range(len(classes))]
        skor_per_kelas.sort(key=lambda x: x[1], reverse=True)

        # Cari skor khusus kelas 'Valid'
        skor_valid = 0.0
        for nama, skor in skor_per_kelas:
            if "valid" in nama.lower():
                skor_valid = skor
                break

        # Ambil daftar khusus kelas Cacat saja (mengabaikan kelas 'Valid')
        cacat_list = [(nama, skor) for nama, skor in skor_per_kelas if "valid" not in nama.lower()]

        # Tampilkan persentase semua kelas
        for nama_kelas, skor in skor_per_kelas:
            if "valid" in nama_kelas.lower():
                st.info(f"Tingkat Keabsahan (**{nama_kelas}**): {skor:.1f}%")
            else:
                st.write(f"- **{nama_kelas}**: {skor:.1f}%")

        st.markdown("---")

        # ==========================================
        # LOGIKA KESIMPULAN BARU
        # ==========================================
        
        # ATURAN 1: Kalau Valid >= 80%, MUTLAK VALID!
        if skor_valid >= 80.0:
            st.success("✅ KESIMPULAN: Wadah ini **NORMAL / VALID** dan siap digunakan!")
            
            # Catatan informasi saja kalau ada indikasi cacat tipis (TIDAK mempengaruhi status Valid)
            cacat_tipis = [f"{nama} ({skor:.1f}%)" for nama, skor in cacat_list if skor >= 20.0]
            if cacat_tipis:
                st.caption(f"ℹ️ *Catatan Indikasi Tipis (Sifatnya hanya informasi): {', '.join(cacat_tipis)}*")

        else:
            # ATURAN 2 & 3: Jika Valid < 80% -> Maka KESIMPULAN ADALAH CACAT / INVALID
            st.error("⚠️ KESIMPULAN: Wadah ini **CACAT / INVALID**!")
            
            # Cek apakah ada cacat yang tembus >= 50%
            cacat_tinggi = [(nama, skor) for nama, skor in cacat_list if skor >= 50.0]

            if len(cacat_tinggi) > 0:
                # ATURAN 2: Tampilkan semua cacat yang tembus >= 50%
                st.write("Alasan Cacat Terdeteksi (Skor Dominan):")
                for nama, skor in cacat_tinggi:
                    st.write(f"🚨 **{nama}** ({skor:.1f}%)")

            else:
                # ATURAN 3: Jika tidak ada cacat yang tembus 50%, ambil Top 2 cacat tertinggi
                top_2_cacat = cacat_list[:2]
                st.write("Alasan Cacat Terdeteksi (Top 2 Dugaan Tertinggi):")
                for nama, skor in top_2_cacat:
                    st.write(f"🚨 **{nama}** ({skor:.1f}%)")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
