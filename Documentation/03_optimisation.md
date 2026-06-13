# 03 — Optimisation & Régularisation — BiLSTM

**Notebook** : `08_optimisation.ipynb`

**Prérequis** : `06_evaluation_comparison.ipynb` exécuté → BiLSTM identifié comme modèle de déploiement (97.87% test accuracy).

**Données** : `client_data.pkl`, `vocab.pkl`, `label_maps.json`

**Sortie** : `../models/bilstm_optimised.pt`

---

## Pourquoi optimiser le BiLSTM et non DistilBERT ?

Les résultats de `06_evaluation_comparison.ipynb` montrent que DistilBERT est le modèle le plus précis (99.50%), devant le BiLSTM (97.87%). Le sujet demande d'optimiser le "meilleur modèle" — ce choix mérite donc une justification.

Le BiLSTM est retenu pour l'optimisation et le déploiement pour deux raisons :

1. **C'est le modèle de production** : 80× plus léger (~3 Mo vs ~250 Mo), 25× plus rapide (~2 ms vs ~50 ms). Dans un contexte de support client à fort volume, ces contraintes priment sur un gain de 2% d'accuracy.
2. **DistilBERT laisse peu de marge** : à 99.50% dès 3 epochs, le fine-tuning est déjà à saturation sur ce dataset. Les techniques d'optimisation classiques (schedulers, grid search, régularisation) n'apporteraient pas de gain mesurable sur un modèle pré-entraîné aussi convergé.

L'optimisation du BiLSTM (97.87% → **98.08% obtenu**) est donc à la fois plus pertinente pédagogiquement et plus cohérente avec le choix de déploiement.

## Objectifs du notebook

L'objectif n'est donc **pas uniquement d'améliorer l'accuracy** mais de :

1. **Démontrer les techniques d'optimisation** sur un modèle réel (pattern TP8)
2. **Analyser la convergence** : est-ce que le modèle peut apprendre plus vite ?
3. **Étudier la régularisation** : gap entre train et val, robustesse au dropout
4. **Trouver les meilleurs hyperparamètres** via Grid/Random Search pour réduire l'écart avec DistilBERT

---

## Structure (pattern TP8)

### Partie 0 — Config & chargement

Même pipeline que notebook 03 : `IntentDataset`, `collate_fn`, `pad_sequence`.

Architecture BiLSTM identique au notebook 03 :
```
Embedding(vocab_size, embed_dim, padding_idx=0)
    ↓
LSTM(embed_dim, hidden_dim, bidirectional=True, num_layers)
    ↓
cat([hidden[-2], hidden[-1]])   → (batch, 2*hidden_dim)
    ↓
Dropout → Linear(2*hidden_dim, 8)
```

### Partie 1 — Baseline (pattern TP8)
*(suivie de la Partie 1b — comparaison d'optimiseurs SGD/Adam/AdamW, voir ci-dessous)*

Ré-entraînement avec les hyperparamètres du notebook 03 — **sans scheduler, sans early stopping**.
Sert de référence : tout gain ultérieur est mesuré par rapport à cette baseline.

| Paramètre | Valeur baseline |
|---|---|
| `EMBED_DIM` | 128 |
| `HIDDEN_DIM` | 256 |
| `DROPOUT` | 0.3 |
| `LR` | 5e-4 |
| `N_EPOCHS` | 15 |
| Optimiseur | Adam |

### Partie 1b — Comparaison d'optimiseurs (SGD vs Adam vs AdamW)

Même architecture et mêmes hyperparamètres que le baseline (`LR=5e-4`, `dropout=0.3`), seul l'optimiseur change. Budget limité à **5 époques** par optimiseur pour comparer la vitesse de convergence et la val_acc atteinte.

| Optimiseur | Principe |
|---|---|
| `SGD` (momentum=0.9) | Descente de gradient classique + momentum |
| `Adam` | Moments adaptatifs (moyenne + variance des gradients) |
| `AdamW` | Comme Adam mais weight decay découplé (`weight_decay=1e-4`) |

**Résultats (5 époques)** :

| Optimiseur | Val Acc (epoch 5) | Best Val Acc |
|---|---|---|
| SGD (momentum=0.9) | 0.7419 | 0.7419 |
| Adam | 0.9764 | 0.9764 |
| AdamW | 0.9763 | 0.9763 |

**Analyse** : SGD+momentum reste très en retard (74.19%) après 5 époques — sa convergence est beaucoup plus lente car le LR=5e-4 (calibré pour Adam) est trop faible pour SGD, qui n'a pas de taux d'apprentissage adaptatif par paramètre. Adam et AdamW convergent quasiment à la même vitesse et atteignent ~97.6% en seulement 5 époques (vs 15 pour le baseline original), confirmant que l'estimation adaptative des moments d'Adam/AdamW est nettement plus efficace que la descente de gradient classique sur ce type de tâche (vocabulaire de 15 000 tokens, embeddings from scratch). AdamW est légèrement en dessous d'Adam ici (0.9763 vs 0.9764) sur seulement 5 époques, l'effet du weight decay découplé (régularisation) se manifestant surtout sur un entraînement plus long — c'est pourquoi **Adam est conservé** pour la suite du notebook (Parties 2 à 7), avec `weight_decay` ajouté séparément dans la Partie 4/5.

### Partie 2 — Schedulers de Learning Rate (pattern TP8)

Deux stratégies comparées :

| Scheduler | Principe | Profil LR |
|---|---|---|
| `ReduceLROnPlateau` | Divise LR par `factor=0.5` si val_loss stagne `patience=2` époques | Escalier |
| `CosineAnnealingLR` | Décroissance cosinus sur `T_max=15` époques | Continu progressif |

> **Pas de LinearWarmup** : ce scheduler est spécifique aux modèles pré-entraînés (BERT).
> Le BiLSTM est entraîné from scratch → un scheduler classique suffit.

Implémentation (pattern TP8) :
```python
if isinstance(scheduler, ReduceLROnPlateau):
    scheduler.step(va_loss)  # prend la métrique
else:
    scheduler.step()         # appel simple après chaque époque
```

### Partie 3 — Early Stopping & sauvegarde (pattern TP8)

Arrête l'entraînement si `val_acc` ne s'améliore pas pendant `patience=4` époques consécutives.
Restaure les poids du **meilleur état** (pas forcément la dernière époque).

```python
if va_acc > best_val_acc:
    best_val_acc = va_acc
    best_weights = copy.deepcopy(model.state_dict())  # snapshot
    no_improve   = 0
else:
    no_improve += 1
    if no_improve >= patience:
        break  # early stopping déclenché

model.load_state_dict(best_weights)  # restauration
```

> **Différence scheduler vs early stopping** :
> - Le scheduler *adapte* le LR pour continuer à apprendre malgré un plateau.
> - L'early stopping *arrête* l'entraînement pour éviter l'overfitting.

### Partie 4 — Régularisation (pattern TP8)

| Technique | Application | Paramètre |
|---|---|---|
| **Dropout** | Entre couches LSTM + avant FC | `dropout` : 0.1 → 0.5 |
| **Weight Decay** | Dans Adam (`weight_decay=1e-4`) | Pénalité L2 sur les poids |
| **num_layers** | Couches LSTM empilées | 1 → 2 |

**Inspection dropout** : vérifié que dropout est actif en mode `train()` et désactivé en `eval()`.

**Rôle de chaque régulariseur** :
- `Dropout` : force le modèle à ne pas dépendre d'un neurone unique → généralisation
- `Weight Decay` : pénalise les poids trop grands → évite l'overfitting sur le vocabulaire
- `num_layers=2` : couche 1 apprend les patterns lexicaux, couche 2 les relations entre patterns

### Partie 5 — Entraînement complet assemblé (pattern TP8)

Combine toutes les techniques :

**LR différencié par groupe** (pattern TP8 Partie 5) :
```python
optim.Adam([
    {'params': model.embedding.parameters(), 'lr': LR * 0.5},  # embedding from scratch
    {'params': model.lstm.parameters(),      'lr': LR},          # LSTM
    {'params': model.fc.parameters(),        'lr': LR * 2}       # tête classification
], weight_decay=1e-4)
```

L'embedding reçoit un LR plus faible (`0.5×`) car il converge plus vite que le LSTM.
Note : le ratio `0.1×` est réservé aux embeddings pré-entraînés (BERT) — ici l'embedding est from scratch, `0.5×` est plus approprié.

**Config finale** :
- `dropout=0.3`, `num_layers=2`
- Scheduler : `CosineAnnealingLR(T_max=20)`
- `patience=5`, `num_epochs=20`

### Partie 6 — Recherche d'hyperparamètres (pattern TP8 Bonus)

**Grid Search** : 9 combinaisons de `(LR, dropout)` — 3 époques par combinaison.

| LR \ Dropout | 0.1 | 0.3 | 0.5 |
|---|---|---|---|
| 1e-3 | 0.9755 | 0.9767 | **0.9770** |
| 5e-4 | 0.9712 | ~0.970 | 0.9682 |
| 1e-4 | 0.9423 | 0.9400 | 0.9356 |

**Conclusion Grid Search** : LR=1e-3 domine sur 3 époques (converge plus vite). LR=1e-4 trop faible, modèle sous-entraîné. L'entraînement complet à LR=5e-4 avec 20 époques reste le meilleur final (98.08%).

**Random Search** : 9 tirages aléatoires dans l'espace continu :
- `LR` : log-uniforme entre 1e-4 et 1e-3
- `dropout` : uniforme entre 0.1 et 0.5

**Pourquoi log-uniforme pour LR ?** Les ordres de grandeur comptent plus que les valeurs absolues :
la différence entre 1e-4 et 5e-4 est plus importante que entre 5e-4 et 9e-4.

**Visualisation** : heatmap Grid Search + scatter plot Random Search (LR en échelle log).

### Partie 7 — Évaluation finale

Évaluation du modèle optimisé (`bilstm_optimised.pt`) sur le test set.
Comparaison avec le BiLSTM du notebook 03.

---

## Hyperparamètres modifiables

| Paramètre | Valeur retenue | Plage explorée | Impact |
|---|---|---|---|
| `LR` | 5e-4 | [1e-4, 1e-3] | Vitesse de convergence |
| `DROPOUT` | 0.3 | [0.1, 0.5] | Régularisation |
| `HIDDEN_DIM` | 256 | [128, 512] | Capacité du modèle |
| `num_layers` | 2 | [1, 2] | Profondeur des abstractions |
| `weight_decay` | 1e-4 | [0, 1e-3] | Pénalité L2 |
| `patience` | 5 | [2, 6] | Seuil early stopping |
| `T_max` | 20 | [10, 20] | Période cosinus |

---

## Résultats obtenus

| Configuration | Best Val Acc |
|---|---|
| Baseline (LR fixe) | 0.9777 |
| ReduceLROnPlateau | 0.9776 |
| CosineAnnealingLR | 0.9779 |
| Early Stopping | 0.9785 |
| Sans régularisation | 0.9779 |
| Dropout + Weight Decay | 0.9786 |
| **Entraînement complet** | **0.9806** |
| **TEST SET** | **0.9808** |

**Gain final** : 97.87% (notebook 03) → 98.08% (+0.21%)

**Analyse par classe** (test set) :

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| acknowledgment | 0.99 | 1.00 | 0.99 |
| complaint | 0.99 | 0.99 | 0.99 |
| follow_up | 0.98 | 0.99 | 0.99 |
| general | 0.98 | 0.95 | 0.97 |
| help_request | 0.99 | 0.99 | 0.99 |
| non_english | 0.97 | 0.99 | 0.98 |
| problem_report | 0.99 | 0.98 | 0.98 |
| question | 0.97 | 0.98 | 0.97 |

**Classes les plus difficiles** :
- `general` (recall 0.95) : 687 tweets mal classifiés — principalement confondus avec `question` (388) et `non_english` (154). Normal : `general` est le catch-all par définition le plus ambigu.
- `question` (precision 0.97) : 55 tweets `problem_report` et 112 tweets `general` prédits à tort comme `question` — tweets contenant `?` sans être de vraies demandes d'info.
- `complaint` → `question` (41 cas) : plaintes formulées sous forme interrogative ("Why is your service so terrible?").