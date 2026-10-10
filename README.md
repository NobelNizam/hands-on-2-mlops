# Hands-on 2 — End-to-End MLflow + scikit-learn

Studi kasus deteksi tumor ganas (Breast Cancer Wisconsin) sesuai alur L1–L9 pada slide. Kelas positif adalah `1 = malignant`; optimizing metric adalah recall dan satisficing metric adalah precision minimal 0,90.

## Struktur

```text
data/                 split train/test yang dilacak DVC
src/data.py           L1: split dan hash data
src/tracking.py       fungsi logging eksperimen standar
src/train.py          L2–L4: baseline, seleksi model, tuning
src/evaluate.py       L5–L6: seleksi otomatis dan final test
src/register.py       L7: registry, alias, quality gate
src/predict.py        L8: prediksi dari @champion
src/reproduce.py      L9: audit lineage dan hash
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc repro prepare
git add .dvc .dvcignore dvc.yaml dvc.lock data/.gitignore
```

Metadata repositori DVC sudah disertakan. Dataset berasal dari scikit-learn sehingga tidak memerlukan unduhan jaringan. DVC menyimpan cache lokal dan Git menyimpan `dvc.lock`, bukan CSV.

## Tracking server

Terminal pertama:

```bash
./start_server.sh
```

Backend metadata tersimpan di `mlflow.db`, artifact di `mlartifacts/`, dan UI tersedia di <http://127.0.0.1:5000>.

## Menjalankan L1–L9

Terminal kedua:

```bash
source .venv/bin/activate
./run_pipeline.sh
```

Atau jalankan tahap satu per satu:

```bash
python -m src.data
python -m src.train
python -m src.evaluate
python -m src.register
python -m src.predict
python -m src.reproduce
```

`src.evaluate` tidak menyentuh test set lagi jika kandidat terpilih sudah memiliki tag `test_evaluated=true`.

## Eksperimen yang dibuat

- Baseline: Dummy, Logistic Regression tanpa scaling, Logistic Regression dengan scaling.
- Model selection: KNN, Decision Tree, Random Forest, Gradient Boosting, dan SVC.
- Tuning: 12 kombinasi `C × gamma` untuk SVC sebagai child runs dari `10-svc-gridsearch`.
- Kandidat dipilih dengan CV recall tertinggi di antara run dengan CV precision ≥ 0,90.
- Model mendapat alias `challenger`; alias `champion` hanya diperbarui bila quality gate lolos.

Di MLflow UI, buka experiment `bc-malignant-detection`, pilih beberapa run, lalu gunakan **Compare** untuk melihat perbedaan parameter dan metrik.

## Serving opsional

```bash
mlflow models serve \
  -m "models:/bc-malignant-classifier@champion" \
  -p 5001 --env-manager local
```

Contoh request REST menggunakan satu record:

```bash
python - <<'PY'
import pandas as pd, requests
X = pd.read_csv("data/test.csv").drop(columns="target").head(1)
payload = {"dataframe_split": X.to_dict(orient="split")}
print(requests.post("http://127.0.0.1:5001/invocations", json=payload, timeout=30).json())
PY
```

## Pemeriksaan

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Set `MLFLOW_TRACKING_URI` bila tracking server tidak memakai alamat default.
