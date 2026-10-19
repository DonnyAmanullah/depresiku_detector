# Panduan Deploy ke Hugging Face Spaces (FastAPI + Docker)

## 1. Ekspor model dari notebook
1. Buka notebook (di Colab/Jupyter yang sama saat training, variabel `X_test_emb` dan `scores_s2` masih ada).
2. Tempel isi `export_model_cell.py` sebagai sel terakhir lalu jalankan. Harus muncul `OK -> [...]`.
3. Unduh folder `deploy/model` (zip dulu di Colab). Isinya: `s1/` (tf_model.h5, config.json, vocab.txt, dll.) dan `s2_params.npz`.
4. Salin isi folder itu ke `depresi-detector/model/`.
5. Cocokkan versi: hasil `print` di sel tadi (tf / transformers) sebaiknya sama dengan `requirements.txt` (tf 2.15.0, transformers 4.36.2). Jika beda, ubah angka di requirements, lalu tetap kunci dengan `==`.

## 2. Uji lokal (opsional, disarankan)
```
python -m venv venv && venv\Scripts\activate      # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```
Buka http://localhost:7860 . Cek http://localhost:7860/api/health harus `{"status":"ok"}`.

## 3. Buat Space
1. huggingface.co → New → Space. Isi nama (mis. `deteksi-depresi`), **SDK: Docker → Blank**, hardware **CPU basic (gratis)**, Public.
2. Buat Access Token: Settings → Access Tokens → New token → role **Write**.

## 4. Upload file
Cara A (paling mudah): tab **Files → Add file → Upload files**, seret seluruh isi folder `depresi-detector` (termasuk `model/`), lalu Commit. File besar otomatis memakai LFS.

Cara B (git):
```
git lfs install
git clone https://huggingface.co/spaces/USERNAME/deteksi-depresi
# salin semua isi depresi-detector ke folder hasil clone
cd deteksi-depresi
git add . && git commit -m "deploy" && git push
```
Password git = Access Token (Write).

## 5. Build & cek log
Tab **App → Logs**: *Build* (instal requirements, 5-10 menit pertama) dan *Container* (jalan aplikasi). Sukses jika ada `Model siap dalam ... detik`.

| Gejala di log | Penyebab & solusi |
|---|---|
| `Gagal memuat model` / `No such file` | Folder `model/s1` atau `model/s2_params.npz` belum terupload |
| `Out of memory` | Pakai hardware CPU basic 16 GB; pastikan hanya 1 worker |
| `ResolutionImpossible` saat build | Ada versi di requirements yang bentrok; samakan dengan versi notebook |
| Halaman "Model belum siap" | Model masih loading (tunggu 1 menit) atau error, lihat Container logs |
| Space "Sleeping" | Normal pada tier gratis, klik untuk membangunkan |

Untuk memulai ulang bersih: Settings → **Factory rebuild**.

## 6. Update aplikasi
Edit file di tab Files (atau git push). Space otomatis rebuild.

## 7. Peta halaman
- `/` landing page (perkenalan aplikasi), `/deteksi` aplikasi deteksi + performa + peneliti, `/api/health` status model, `/api/predict` endpoint prediksi.
- Jika Space sudah terlanjur dibuat, cukup upload ulang: `app/main.py`, `app/templates/landing.html`, `app/templates/index.html`, `app/static/style.css`. Space akan rebuild otomatis.
