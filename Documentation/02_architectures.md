# 02 — Architectures de modèles

**Prérequis** : `01_clustering.ipynb` exécuté — artefacts `client_data.pkl`, `client_text_data.pkl`, `vocab.pkl`, `label_maps.json` présents dans `../data/`.

---

## Vue d'ensemble

| # | Architecture | Notebook | Données | Pattern TP |
|---|---|---|---|---|
| 02 | RNN | `02_model_rnn.ipynb` | `client_data.pkl` | TP5 |
| 03 | LSTM + BiLSTM | `03_model_lstm.ipynb` | `client_data.pkl` | TP6 |
| 04 | Transformer from scratch | `04_model_transformer.ipynb` | `client_data.pkl` | TP7 |
| 05 | DistilBERT fine-tuné | `05_model_distilbert.ipynb` | `client_text_data.pkl` | — |
| 06 | Comparaison | `06_evaluation_comparison.ipynb` | résultats `.pkl` | — |
| 07 | CNN + BiLSTM (hybride) | `07_model_cnn_lstm.ipynb` | `client_data.pkl` | TP4 + TP6 |

**8 classes d'intention** : `acknowledgment`, `complaint`, `follow_up`, `general`, `help_request`, `non_english`, `problem_report`, `question`.

**Résultats finaux** (données issues de `01_clustering.ipynb`, labels keyword-based) :

| Modèle | Test Accuracy | Best Val | Epochs |
|---|---|---|---|
| **DistilBERT** | **99.50%** | 99.53% | 3 |
| CNN+BiLSTM | 97.92% | 97.86% | 15 |
| BiLSTM | 97.87% | 97.76% | 15 |
| LSTM | 97.83% | 97.81% | 15 |
| Transformer | 96.71% | 96.73% | 10 |
| RNN | 95.66% | 95.65% | 10 |

> **Note sur les résultats** : DistilBERT se détache clairement avec ~2% d'écart sur les modèles from scratch. Cet écart reflète la valeur du pré-entraînement sur 11 milliards de tokens — ses représentations contextuelles capturent des nuances que les embeddings custom (vocabulaire 15 000 tokens, entraînés from scratch) ne peuvent pas atteindre. Les labels étant keyword-based, les modèles from scratch convergent autour de 97-98% en apprenant les patterns lexicaux directs ; DistilBERT dépasse cette limite grâce à sa compréhension contextuelle pré-apprise. Le BiLSTM reste le choix retenu pour le déploiement pour des raisons de taille et de latence (voir `04_deploiement.md`).

---

## Composants partagés (notebooks 02, 03, 04, 07)

### Données d'entrée

Artefacts produits par `01_clustering.ipynb` :

| Fichier | Contenu | Utilisé par |
|---|---|---|
| `vocab.pkl` | `{token: id}`, 15 000 entrées | 02, 03, 04, 07 |
| `label_maps.json` | `{id: label_str}` | tous |
| `client_data.pkl` | `{train/val/test: [(token_ids, label_int)]}` | 02, 03, 04, 07 |
| `client_text_data.pkl` | `{train/val/test: [(text_str, label_int)]}` | 05 uniquement |

### Dataset + DataLoader (pattern TP5)

```python
class IntentDataset(Dataset):
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence([torch.tensor(t, dtype=torch.long) for t in texts],
                                 batch_first=True, padding_value=PAD_IDX)
    return texts_padded, torch.tensor(labels, dtype=torch.long)
```

`pad_sequence` uniformise les longueurs dans chaque batch en ajoutant des `<PAD>` (id=0) à droite. Le paramètre `batch_first=True` donne la forme `(batch, seq_len)` attendue par tous les modèles.

### Boucle d'entraînement commune (pattern TP5/TP6)

```
Pour chaque époque :
  1. model.train() → passe forward + backward sur train
  2. model.eval()  → passe forward sur val, no_grad
  3. Sauvegarder le meilleur état si val_acc s'améliore
Après entraînement :
  Charger le meilleur état → évaluation sur test
```

### Évaluation commune

- `accuracy_score` globale
- `classification_report` par classe (precision, recall, F1)
- `confusion_matrix` heatmap seaborn
- Courbes train/val loss et accuracy sauvegardées en PNG

---

## Architecture 1 — RNN (`02_model_rnn.ipynb`)

### Principe (TP5)

Le RNN traite la séquence **token par token**. À chaque pas `t`, il met à jour un état caché qui résume tout ce qui a été lu :

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**Limite** : lors de la rétropropagation, les gradients sont multipliés à travers tous les pas de temps. Sur des séquences longues, ce produit tend vers 0 — c'est le **vanishing gradient** (TP5). Le modèle n'apprend plus les dépendances du début de séquence.

**Choix de pooling** : le notebook utilise `output.mean(dim=1)` (moyenne sur tous les états cachés) au lieu de `hidden[-1]` (TP5). Sur 8 classes, le mean pooling inclut les mots du début de séquence que `hidden[-1]` ne capte plus à cause du vanishing gradient.

### Structure

```
x (batch, seq_len)
    ↓  Embedding(vocab_size, embed_dim, padding_idx=0)
(batch, seq_len, embed_dim)
    ↓  RNN(embed_dim, hidden_dim, batch_first=True)
output : (batch, seq_len, hidden_dim)
    ↓  output.mean(dim=1)
(batch, hidden_dim)
    ↓  Dropout → Linear(hidden_dim, 8)
(batch, 10)
```

### Paramètres et hyperparamètres

| Paramètre | Valeur actuelle | Ajustable | Impact si modifié |
|---|---|---|---|
| `EMBED_DIM` | 128 | ✅ | 64 → under-représentation sur 8 classes ; 256 → gain marginal sur tweets courts |
| `HIDDEN_DIM` | 256 | ✅ | 128 → moins de capacité ; 512 → overfitting |
| `DROPOUT` | 0.3 | ✅ | Augmenter si val_loss remonte avant train_loss |
| `LR` | 1e-3 | ✅ | Standard Adam ; réduire à 5e-4 si oscillations |
| `BATCH_SIZE` | 64 | ✅ | 128 → gradient plus lisse mais moins de mises à jour |
| `N_EPOCHS` | 10 | ✅ | Le meilleur modèle est sauvegardé automatiquement |
| `clip_grad_norm_` | max_norm=1.0 | ✅ | Évite l'explosion de gradient — spécifique au RNN |
| `output.mean(dim=1)` | — | ⚠️ | Ne pas remplacer par `hidden[-1]` : vanishing gradient sur 64 tokens |

---

## Architecture 2 — LSTM + BiLSTM (`03_model_lstm.ipynb`)

### Principe (TP6)

Le LSTM résout le vanishing gradient avec un **cell state** `c_t` (mémoire long terme) et trois portes :

```
f_t = σ(W_f · [h_{t-1}, x_t])   # Forget gate — quoi oublier de c_{t-1}
i_t = σ(W_i · [h_{t-1}, x_t])   # Input gate  — quoi ajouter à c_t
g_t = tanh(W_g · [h_{t-1}, x_t])# Candidate cell
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t # Addition (pas multiplication) → gradient préservé
o_t = σ(W_o · [h_{t-1}, x_t])   # Output gate
h_t = o_t ⊙ tanh(c_t)
```

La mise à jour par **addition** est la clé : le gradient remonte sans s'atténuer (*constant error carousel*, TP6).

**BiLSTM** : deux LSTM traitent la séquence en sens opposés. Les deux derniers états cachés sont concaténés :
```python
last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 2*hidden_dim)
```
Le modèle dispose ainsi du contexte passé ET futur pour chaque token.

### Structure LSTM

```
x (batch, seq_len)
    ↓  Embedding
    ↓  LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
hidden[-1] : (batch, hidden_dim)
    ↓  Dropout → Linear(hidden_dim, 8)
```

### Structure BiLSTM

```
x (batch, seq_len)
    ↓  Embedding
    ↓  LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
cat([hidden[-2], hidden[-1]]) : (batch, 2*hidden_dim)
    ↓  Dropout → Linear(2*hidden_dim, 8)
```

### Paramètres et hyperparamètres

| Paramètre | Valeur actuelle | Ajustable | Impact si modifié |
|---|---|---|---|
| `EMBED_DIM` | 128 | ✅ | Idem RNN |
| `HIDDEN_DIM` | 256 | ✅ | BiLSTM : sortie = 2×256 = 512 ; 128 → trop peu sur 8 classes |
| `NUM_LAYERS` | 2 | ✅ | 1 → moins de capacité ; 3+ → overfitting sur tweets courts |
| `DROPOUT` | 0.3 | ✅ | Appliqué entre couches LSTM (si `num_layers > 1`) ET avant FC |
| `LR` | 5e-4 | ✅ | Plus faible que RNN : LSTM multi-couches sensible aux grands pas |
| `BATCH_SIZE` | 64 | ✅ | — |
| `N_EPOCHS` | 15 | ✅ | LSTM converge plus lentement que RNN (4× plus de paramètres par gate) |

---

## Architecture 3 — Transformer from scratch (`04_model_transformer.ipynb`)

### Principe (TP7)

Le Transformer traite tous les tokens **en parallèle** via l'attention. Il ne connaît pas l'ordre naturellement → **encodage positionnel** sinusoïdal ajouté aux embeddings.

#### Encodage positionnel (TP7)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Chaque position a une signature unique. Les fréquences varient selon la dimension : basses dimensions oscillent vite (positions proches distinguées), hautes dimensions oscillent lentement (positions éloignées distinguées).

#### Multi-Head Self-Attention (TP7)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- **Q** (Queries) : ce que le token cherche
- **K** (Keys) : ce que chaque token contient  
- **V** (Values) : ce que chaque token transmet
- `1/√d_k` : évite la saturation du softmax

`num_heads` têtes en parallèle → chaque tête apprend à regarder des relations différentes.

Un **masque de padding** `(x == 0)` est passé à l'attention pour ignorer les tokens `<PAD>`.

#### TransformerBlock (TP7)

```
x → LayerNorm → MultiHeadSelfAttention → Dropout → (+x)   résidu 1
  → LayerNorm → FFN(Linear→ReLU→Linear) → Dropout → (+x)  résidu 2
```

Les connexions résiduelles permettent au gradient de circuler sans s'atténuer.

#### Pooling

Pas d'état final naturel → **moyenne masquée** sur les tokens non-PAD :
```python
mask_expanded = (~padding_mask).unsqueeze(-1).float()
pooled = (emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
```

### Structure

```
x (batch, seq_len)
    ↓  Embedding × √embed_dim
    ↓  PositionalEncoding
    ↓  Dropout
    ↓  N × TransformerBlock(embed_dim, num_heads, ff_dim)
(batch, seq_len, embed_dim)
    ↓  Masked mean pooling
(batch, embed_dim)
    ↓  Linear(embed_dim, 8)
```

### Paramètres et hyperparamètres

| Paramètre | Valeur actuelle | Ajustable | Impact si modifié |
|---|---|---|---|
| `EMBED_DIM` | 128 | ✅ | **Doit être divisible par `NUM_HEADS`** : 128/4 = 32 dims/tête |
| `NUM_HEADS` | 4 | ✅ | 8 têtes → `EMBED_DIM` doit être ≥ 256 ; plus de têtes = relations plus variées |
| `FF_DIM` | 256 | ✅ | Conventionnellement 2-4× `EMBED_DIM` ; papier original utilise 4× |
| `NUM_BLOCKS` | 2 | ✅ | Plus de blocs → plus profond, mais overfitting from scratch sur tweets courts |
| `DROPOUT` | 0.1 | ✅ | Plus faible que LSTM : les résidus régularisent déjà ; ne pas dépasser 0.3 |
| `MAX_LEN` | 64 | ✅ | Tweets : moyenne 17 tokens, P95 ≈ 40 → 64 couvre 99%+ |
| `LR` | 1e-3 | ✅ | Avec un scheduler warmup → meilleure convergence |
| `N_EPOCHS` | 10 | ✅ | Convergence rapide sur tweets courts |

---

## Architecture 4 — DistilBERT fine-tuné (`05_model_distilbert.ipynb`)

### Principe

Contrairement aux 4 architectures précédentes qui apprennent les représentations from scratch sur ~134 000 tweets, **DistilBERT** est pré-entraîné sur 11 milliards de tokens (Wikipedia + BooksCorpus). Ses embeddings contextuels distinguent "bank" dans "river bank" et "bank account" — impossible avec un vocabulaire custom de 15 000 tokens.

Le **fine-tuning** continue l'entraînement sur notre tâche avec un LR très faible pour spécialiser le modèle sans détruire les représentations pré-apprises.

**DistilBERT vs BERT** : version distillée de BERT, 6 couches (vs 12), 66M paramètres (vs 110M), préserve 97% des capacités avec 40% moins de paramètres.

**Tokenizer WordPiece** : décompose les mots rares en sous-mots (`"cancellation"` → `["cancel", "##lation"]`). Pas de `<UNK>` pour les noms de marques ou abréviations Twitter — avantage direct sur notre vocab custom.

### Structure

```
Texte brut
    ↓  WordPiece tokenizer (vocab ≈ 30 000) + [CLS]/[SEP]
input_ids, attention_mask : (batch, 128)
    ↓  DistilBERT encoder × 6 blocs Transformer (embed_dim=768, heads=12, ff=3072)
    ↓  pre_classifier : Linear(768, 768) → ReLU → Dropout
    ↓  classifier : Linear(768, 8)
(batch, 10)
```

Le token `[CLS]` agrège l'information de toute la séquence et sert d'entrée à la tête de classification.

### Paramètres et hyperparamètres

| Paramètre | Valeur actuelle | Ajustable | Impact si modifié |
|---|---|---|---|
| `MAX_LEN` | 128 | ✅ | 64 → tronque certains tweets (WordPiece allonge) ; 256 → lent, inutile |
| `BATCH_SIZE` | 32 | ✅ | 64 → OOM GPU probable ; 16 → entraînement lent |
| `LR` | 2e-5 | ✅ | Plage standard BERT : 1e-5 à 5e-5 ; > 5e-5 → catastrophic forgetting |
| `N_EPOCHS` | 3 | ✅ | 3-5 suffisent ; au-delà → overfitting quasi-certain |
| `weight_decay` | 0.01 | ✅ | Dans AdamW : régularisation L2 sur les poids (pas les biais) |
| `clip_grad_norm_` | max_norm=1.0 | ✅ | Standard fine-tuning BERT |
| `optimizer` | AdamW | ⚠️ | Ne pas remplacer par Adam standard : AdamW découple le weight decay |
| `class_weights` | non utilisé | ⚠️ | Sur modèle pré-entraîné : cause sur-prédiction des classes rares |

---

## Architecture 5 — CNN + BiLSTM (`07_model_cnn_lstm.ipynb`)

### Principe (TP4 + TP6)

Les architectures précédentes traitent les embeddings token par token. Le CNN + BiLSTM **combine deux étapes** :

1. **CNN 1D** (TP4) : filtre glissant de taille `kernel_size` → détecteur de n-grammes appris automatiquement. Chaque filtre détecte un pattern local différent ("flight cancelled", "not working", "please refund").

2. **BiLSTM** (TP6) : reçoit les features CNN (non plus des embeddings bruts) → modélise les relations entre ces patterns locaux.

```
CNN → transforme les embeddings en features de n-grammes
BiLSTM → modélise les dépendances entre ces features
```

**Avantage mécanique** : le CNN réduit la longueur de séquence (64 → 62 pour `kernel=3`), ce qui atténue le vanishing gradient du BiLSTM. Chaque pas du BiLSTM représente déjà un contexte de 3 tokens.

**Pourquoi cette combinaison est particulièrement adaptée à la classification d'intention sur tweets** : les intentions se manifestent souvent par des expressions locales et fixes — "not working", "please help", "still waiting", "flight cancelled". Le CNN détecte ces n-grammes caractéristiques mieux qu'un LSTM seul, qui doit les reconstruire token par token depuis le début de la séquence. Le BiLSTM prend ensuite ces signaux locaux comme entrée et modélise leurs relations dans la phrase — par exemple, distinguer "still waiting for help" (follow_up) de "still waiting, terrible service" (complaint). Les deux architectures se complètent : l'une pour la reconnaissance de patterns locaux, l'autre pour le contexte global.

### Structure

```
x (batch, seq_len)
    ↓  Embedding(vocab_size, embed_dim, padding_idx=0)
(batch, seq_len, embed_dim)
    ↓  permute(0,2,1)
(batch, embed_dim, seq_len)          ← Conv1D attend (batch, channels, length)
    ↓  Conv1d(embed_dim, num_filters, kernel_size) → ReLU
(batch, num_filters, seq_len - kernel_size + 1)
    ↓  permute(0,2,1)
(batch, seq_len', num_filters)       ← séquence de features locales
    ↓  LSTM(num_filters, hidden_dim, bidirectional=True)
cat([hidden[-2], hidden[-1]]) : (batch, 2*hidden_dim)
    ↓  Dropout → Linear(2*hidden_dim, 8)
```

### Paramètres et hyperparamètres

| Paramètre | Valeur actuelle | Ajustable | Impact si modifié |
|---|---|---|---|
| `EMBED_DIM` | 128 | ✅ | Idem autres modèles |
| `NUM_FILTERS` | 256 | ✅ | Nombre de patterns CNN appris ; 128 → sous-représentation ; 512 → lent |
| `KERNEL_SIZE` | 3 | ✅ | Trigrammes optimal sur tweets (~17 tokens) ; 2 = bigrammes trop courts ; 5 = rare dans tweets |
| `HIDDEN_DIM` | 256 | ✅ | BiLSTM reçoit `num_filters` en entrée → garder proche de `NUM_FILTERS` |
| `NUM_LAYERS` | 2 | ✅ | Idem LSTM |
| `DROPOUT` | 0.3 | ✅ | Appliqué entre couches LSTM ET avant FC |
| `LR` | 5e-4 | ✅ | Plus faible que 1e-3 : les features CNN amplifient les gradients |
| `BATCH_SIZE` | 128 | ✅ | Plus grand que les autres modèles : CNN + BiLSTM supporte mieux les grands batchs |
| `N_EPOCHS` | 15 | ✅ | Avec early stopping — arrêt automatique |
| `PATIENCE` | 4 | ✅ | Nombre d'époques sans amélioration avant arrêt |
| `scheduler` | `ReduceLROnPlateau(factor=0.5, patience=2)` | ✅ | Divise LR par 2 si val_loss stagne — affine sans schedule fixe |

---

## Notebook 06 — Comparaison (`06_evaluation_comparison.ipynb`)

Charge tous les fichiers `results_*.pkl` générés par les notebooks 02-07 et produit :

1. **Bar chart** des test accuracies (bleu = from scratch, vert = hybride, orange = pré-entraîné)
2. **Courbes val loss comparées** sur toutes les époques
3. **Matrices de confusion** côte à côte
4. **Tableau récapitulatif** classé par test accuracy

---

## Synthèse des hyperparamètres ajustables

Récapitulatif pour l'étape d'optimisation sur le meilleur modèle :

| Hyperparamètre | RNN | LSTM/BiLSTM | Transformer | CNN+BiLSTM | DistilBERT |
|---|---|---|---|---|---|
| `EMBED_DIM` | 128 | 128 | 128 | 128 | fixé (768) |
| `HIDDEN_DIM` | 256 | 256 | — | 256 | fixé |
| `NUM_LAYERS` | — | 2 | — | 2 | fixé (6) |
| `NUM_HEADS` | — | — | 4 | — | fixé (12) |
| `FF_DIM` | — | — | 256 | — | fixé (3072) |
| `NUM_BLOCKS` | — | — | 2 | — | fixé |
| `NUM_FILTERS` | — | — | — | 256 | — |
| `KERNEL_SIZE` | — | — | — | 3 | — |
| `DROPOUT` | 0.3 | 0.3 | 0.1 | 0.3 | via AdamW |
| `LR` | 1e-3 | 5e-4 | 1e-3 | 5e-4 | 2e-5 |
| `BATCH_SIZE` | 64 | 64 | 64 | 128 | 32 |
| `N_EPOCHS` | 10 | 15 | 10 | 15 (+ES) | 3 |
| `weight_decay` | — | — | — | — | 0.01 |
| `MAX_LEN` | 64 | 64 | 64 | 64 | 128 |