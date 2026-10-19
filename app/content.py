"""Data statis dari paper (angka diambil apa adanya dari naskah)."""
PAPER = {
    "title": "Depression Detection on Indonesian Social Media Using Fine-Tuned IndoBERT and SVM",
    "journal": "Sinkron : Jurnal dan Penelitian Teknik Informatika",
    "volume": "Volume 10, Number 3, July 2026",
    "issn": "e-ISSN: 2541-2019, p-ISSN: 2541-044X",
    "doi": "10.33395/sinkron.v10i3.16294",
    "doi_url": "https://doi.org/10.33395/sinkron.v10i3.16294",
    "url": "https://www.jurnal.polgan.ac.id/index.php/sinkron/article/view/16294",
    "affiliation": "Departemen Teknik Informatika, Fakultas Ilmu Komputer, Universitas Dian Nuswantoro, Semarang, Indonesia",
    "authors": [
        ("Donny Amanullah Putra Rahman", "Penulis korespondensi", "111202315426@mhs.dinus.ac.id"),
        ("Dr. Muhamad Akrom, S.Si, M.Si.", "Penulis", "m.akrom@dsn.dinus.ac.id"),
        ("Muhammad Naufal, S.Tr.T, M.Kom", "Penulis", "m.naufal@dsn.dinus.ac.id"),
    ],
    "stats": [("10.082", "tweet mentah"), ("3.874", "tweet unik dipakai"), ("73,71%", "akurasi tertinggi"), ("0,8212", "AUC-ROC tertinggi")],
    "models": {
        "s1": dict(name="S1 · IndoBERT Fine-Tuned", acc="73,20", auc="0,8212", f1="0,730", size="475,24 MB",
                   dep=("0,690", "0,730", "0,710"), non=("0,770", "0,734", "0,751"), cm=[[157, 57], [47, 127]]),
        "s2": dict(name="S2 · IndoBERT + SVM", acc="73,71", auc="0,8088", f1="0,735", size="0,01 MB*",
                   dep=("0,700", "0,724", "0,712"), non=("0,769", "0,748", "0,758"), cm=[[160, 54], [48, 126]]),
    },
    "baselines": [
        ("S2 · IndoBERT + SVM", 73.71, "0,8088", "0,735", True),
        ("S1 · IndoBERT Fine-Tuned", 73.20, "0,8212", "0,730", True),
        ("TF-IDF + SVM (linear)", 70.36, "0,7828", "0,6999", False),
        ("TF-IDF + Naive Bayes", 70.36, "0,7750", "0,6933", False),
        ("TF-IDF + Gradient Boosting", 67.53, "0,6956", "0,6673", False),
        ("TF-IDF + Random Forest", 66.49, "0,7368", "0,6613", False),
        ("TF-IDF + Decision Tree", 57.22, "0,5697", "0,5692", False),
    ],
    "steps": [
        ("Data", "10.082 tweet berbahasa Indonesia (Budiman, 2021), dilabeli awal oleh psikolog, dideduplikasi menjadi 3.874."),
        ("Pembagian", "Stratified 80:10:10 (latih 3.099, validasi 387, uji 388) tanpa tumpang-tindih."),
        ("Fine-tuning", "indobert-base-p1, 4 blok encoder teratas dilatih, AdamW 3×10⁻⁵, early stopping."),
        ("Klasifikasi", "S1: kepala klasifikasi softmax. S2: embedding [CLS] 768-dim, StandardScaler, lalu Linear SVM."),
    ],
}
