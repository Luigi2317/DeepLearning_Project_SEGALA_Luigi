# Classification d'intentions client — Twitter Customer Support

Projet Deep Learning : classification automatique des intentions dans les messages clients Twitter en 8 classes (`acknowledgment`, `complaint`, `follow_up`, `general`, `help_request`, `non_english`, `problem_report`, `question`).

**Dataset** : [Twitter Customer Support](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~3M tweets, Kaggle)  
**Modèle en production** : BiLSTM optimisé (98.08% accuracy, ~16 Mo, ~3 ms/tweet)

---

## Structure du projet

```
notebooks/          9 notebooks d'expérimentation (ordre d'exécution 01→09)
models/             Modèles entraînés sauvegardés (.pt) — générés en exécutant les notebooks
data/               Données prétraitées (client_data.pkl, vocab.pkl, ...) — générées en exécutant les notebooks
exports/            Modèles exportés (TorchScript, ONNX, quantizé) — générés par 09_inference_deploiement.ipynb
api_bilstm.py       API FastAPI de déploiement
requirements.txt    Dépendances Python
Rapport_Deep_Learning_SEGALA_Luigi.pdf/.docx   Rapport technique complet
demo_api.mp4        Démonstration vidéo de l'API
```

> **Note** : `data/`, `models/` et `exports/` ne sont pas versionnés (trop volumineux) — ils sont générés en exécutant les notebooks dans l'ordre ci-dessous.

---

## Lancer le projet — ordre d'exécution

Les notebooks se lancent dans l'ordre numérique. Chaque notebook charge les fichiers produits par le précédent.

| Notebook | Rôle | Prérequis | Produit |
|---|---|---|---|
| `01_clustering.ipynb` | Exploration, nettoyage, labellisation, split | `twcs.csv` | `client_data.pkl`, `vocab.pkl`, `label_maps.json`, `client_text_data.pkl` |
| `02_model_rnn.ipynb` | RNN simple — baseline | `client_data.pkl` | `rnn_model.pt` |
| `03_model_lstm.ipynb` | LSTM & BiLSTM | `client_data.pkl` | `bilstm_model.pt` |
| `04_model_transformer.ipynb` | Transformer from scratch | `client_data.pkl` | `transformer_model.pt` |
| `05_model_distilbert.ipynb` | Fine-tuning DistilBERT | `client_text_data.pkl` | `distilbert_model.pt` |
| `06_evaluation_comparison.ipynb` | Comparaison des 6 architectures | tous les `.pt` | tableaux + graphiques |
| `07_model_cnn_lstm.ipynb` | CNN + BiLSTM hybride | `client_data.pkl` | `cnn_bilstm_model.pt` |
| `08_optimisation.ipynb` | Optimisation BiLSTM (schedulers, grid search) | `client_data.pkl` | `bilstm_optimised.pt` |
| `09_inference_deploiement.ipynb` | Export TorchScript/ONNX, quantization, benchmark | `bilstm_optimised.pt` | `exports/` |

**Remarque** : `01_clustering.ipynb` requiert un GPU pour les embeddings cardiffnlp (~3 min sur P100). Les notebooks 02-08 tournent sur CPU ou GPU.

---

## Lancer l'API FastAPI

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API (nécessite bilstm_optimised.pt, vocab.pkl, label_maps.json)
uvicorn api_bilstm:app --reload --port 8000
```

L'API expose deux endpoints :

- `POST /predict` — classifie un tweet
- `GET /health` — vérifie que l'API est opérationnelle

**Exemple de requête** :
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "my phone keeps crashing since the update"}'
```

**Réponse** :
```json
{"intention": "problem_report", "confiance": 0.99, "top3": [...]}
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Démonstration

`demo_api.mp4` — courte vidéo montrant l'API déployée répondant à plusieurs requêtes `/predict` via Swagger UI.

---

## Résultats — comparaison des 6 architectures

| Modèle | Test Accuracy | Paramètres | Latence CPU |
|---|---|---|---|
| RNN | 95.66% | ~2M | ~1 ms |
| LSTM | 97.83% | ~4M | ~1.5 ms |
| BiLSTM | 97.87% | ~6M | ~2 ms |
| Transformer (scratch) | 96.71% | ~5M | ~3 ms |
| CNN + BiLSTM | 97.92% | ~5M | ~2 ms |
| DistilBERT | **99.50%** | ~66M | ~50 ms |
| **BiLSTM optimisé** | **98.08%** | ~6M | ~2 ms |

**Modèle retenu pour le déploiement** : BiLSTM optimisé — 15× plus léger que DistilBERT (~16 Mo vs 268 Mo), sans dépendance HuggingFace en production.
