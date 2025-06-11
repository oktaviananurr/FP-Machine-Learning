
import os
import cv2
from yolov5.detect import run  # pastikan ini mengarah ke file detect.py dari YOLOv5
import streamlit as st
import numpy as np
import tempfile
from glob import glob


# ✅ INI HARUS PALING ATAS setelah import
st.set_page_config(page_title="Deteksi Gangguan Kulit", layout="wide")

# Daftar nama kelas (ganti sesuai kelas yang kamu pakai)
class_names = [
    'Darier_s_Disease',
    'Epidermolysis_Bullosa_Pruriginosa',
    'Hailey_Hailey_Disease',
    'Hemangiome',
    'Impetigo',
    'Leishmanios',
    'Lupus_Erythematosus_Chronicus_Discoides',
    'Melanoma',
    'Molluscum_Contagiosum',
    'Porokeratosis',
    'Psoriasis',
    'Tinea_Corporis',
    'Tungiasis',
    'acne',
    'basal_cell_carcinoma',
    'eczema',
    'lichen',
    'nevus',
    'normal_skin']

def load_image_opencv(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def get_latest_exp_folder(base_path="yolov5/runs/detect"):
    exp_folders = sorted(glob(os.path.join(base_path, "exp*")), key=os.path.getmtime)
    return exp_folders[-1] if exp_folders else None

def read_detection_labels(txt_path):
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            labels = [int(line.split()[0]) for line in f.readlines()]
    return labels

def run_detection_logic(file):
    image = load_image_opencv(file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    result = {"filename": file.name, "labels": [], "image_path": None, "error": None}

    try:
        run(
            weights='/workspaces/FP-Machine-Learning/New-Mod-1/best.pt',
            source=tmp_path,
            conf_thres=0.3,
            iou_thres=0.45,
            imgsz=(640, 640),
            save_txt=True,
            save_conf=True,
            save_crop=False,
            project='yolov5/runs/detect',
            name='exp',
            exist_ok=True
        )

        latest_exp = get_latest_exp_folder("yolov5/runs/detect")

        if latest_exp:
            img_name = os.path.basename(tmp_path)
            img_path = os.path.join(latest_exp, img_name)
            txt_path = os.path.join(latest_exp, "labels", os.path.splitext(img_name)[0] + ".txt")

            # Cek fallback kalau image tidak ditemukan langsung
            if not os.path.exists(img_path):
                possible_images = glob(os.path.join(latest_exp, os.path.splitext(img_name)[0] + ".*"))
                img_path = possible_images[0] if possible_images else None
                if not img_path:
                    result["error"] = f"Gambar hasil deteksi tidak ditemukan di {latest_exp}"

            labels = read_detection_labels(txt_path)
            label_names = [class_names[c] for c in labels if c < len(class_names)]

            result.update({
                "labels": label_names,
                "image_path": img_path if img_path and os.path.exists(img_path) else None
            })

            if img_path and not os.path.exists(img_path):
                result["error"] = f"Gambar hasil deteksi ({img_path}) tidak ditemukan."

        else:
            result["error"] = "❌ Folder hasil deteksi (exp*) tidak ditemukan di 'yolov5/runs/detect'."

    except Exception as e:
        st.error(f"⚠️ Deteksi gagal untuk {file.name}: {e}")
        result["error"] = str(e)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


# --- STREAMLIT ---
st.title("🩺 Deteksi Ganguan Kulit Otomatis")
st.markdown("Unggah gambar kulit, lalu tekan *Mulai Deteksi*.")

if 'detection_results' not in st.session_state:
    st.session_state['detection_results'] = []
if 'uploaded_file_key' not in st.session_state:
    st.session_state['uploaded_file_key'] = 0

uploaded_file = st.file_uploader("Upload gambar kulit", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state['uploaded_file_key']}")

col1, col2 = st.columns(2)
with col1:
    detect_btn = st.button("🔍 Mulai Deteksi")
with col2:
    clear_btn = st.button("🗑 Hapus Gambar")

if detect_btn:
    if uploaded_file is None:
        st.warning("⚠ Harap upload gambar terlebih dahulu.")
    else:
        with st.spinner("🔎 Mendeteksi..."):
            result = run_detection_logic(uploaded_file)
            st.session_state['detection_results'] = [result]

if clear_btn:
    st.session_state['detection_results'] = []
    st.session_state['uploaded_file_key'] += 1
    st.rerun()

# --- TAMPILKAN HASIL ---
if st.session_state['detection_results']:
    for result in st.session_state['detection_results']:
        st.subheader(f"Hasil Deteksi untuk: {result['filename']}")
        if result['image_path'] and os.path.exists(result["image_path"]):
            st.image(result["image_path"], caption="Deteksi Penyakit", use_container_width=True)

        if result['labels']:
            counts = Counter(result["labels"])
            st.success("✅ Penyakit yang Terdeteksi:")
            for label, count in counts.items():
                st.markdown(f"- *{label.replace('_', ' ').title()}* (jumlah: {count})")
        else:
            st.info("⚠ Tidak ada penyakit kulit terdeteksi atau deteksi di bawah ambang batas.")

elif not uploaded_file: # Initial state or after clearing
    st.info("Selamat datang! Silakan upload gambar kulit yang ingin diperiksa dan klik 'Jalankan Deteksi'.")




