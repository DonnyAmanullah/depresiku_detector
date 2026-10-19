# TEMPEL & JALANKAN sebagai sel TERAKHIR di notebook (setelah sel Scenario 1 & 2 selesai).
import os, shutil, numpy as np, joblib, sklearn, transformers, tensorflow as tf
print("tf", tf.__version__, "| transformers", transformers.__version__, "| sklearn", sklearn.__version__)
os.makedirs("deploy/model", exist_ok=True)
shutil.copytree("S1_FineTuned_IndoBERT", "deploy/model/s1", dirs_exist_ok=True)
svm = joblib.load("S2_FineTuned_SVM_model.pkl"); sc = joblib.load("S2_scaler.pkl")
# Simpan SVM + scaler sebagai array biasa -> app tidak butuh scikit-learn (bebas masalah versi pickle)
np.savez("deploy/model/s2_params.npz", coef=svm.coef_, intercept=svm.intercept_, mean=sc.mean_, scale=sc.scale_)
# Uji kebenaran: rumus manual harus sama dengan decision_function di notebook
assert np.allclose(X_test_emb @ svm.coef_.ravel() + svm.intercept_[0], scores_s2, atol=1e-5), "Rumus S2 tidak cocok!"
print("OK ->", os.listdir("deploy/model"), os.listdir("deploy/model/s1"))
