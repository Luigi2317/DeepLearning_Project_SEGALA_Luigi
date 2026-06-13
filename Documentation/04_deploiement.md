# 04 — Export, Inférence et Déploiement — BiLSTM

**Notebook** : `09_inference_deploiement.ipynb`

**Prérequis** : `08_optimisation.ipynb` exécuté → `bilstm_optimised.pt` disponible.

**Sortie** : `../exports/` (TorchScript, ONNX, modèle quantizé) + `api_bilstm.py`

---

## Objectif

Préparer le BiLSTM pour un déploiement en production (pattern TP10) :
1. Exporter en formats portables (TorchScript, ONNX)
2. Réduire la taille et accélérer l'inférence (quantization)
3. Benchmarker les 3 formats
4. Créer une fonction d'inférence sur texte brut
5. Exposer via une API REST FastAPI

---

## Structure (pattern TP10)

### Partie 0 — Rechargement du modèle

```python
# map_location='cpu' : nécessaire pour déploiement sans GPU et pour TorchScript/ONNX
model.load_state_dict(torch.load('bilstm_optimised.pt', map_location='cpu'))
model.eval()
```

**Accuracy de référence** : évaluée sur le test set avant export pour comparaison.

### Partie 1 — TorchScript (pattern TP10)

| Méthode | Principe | Pour le BiLSTM |
|---|---|---|
| `torch.jit.trace` | Suit l'exécution sur un exemple, fige les opérations | ✅ Architecture statique |
| `torch.jit.script` | Analyse le code Python complet | Non nécessaire ici |

Le BiLSTM est une architecture **statique** (pas de branches conditionnelles dans `forward`) → `trace` suffit.

```python
dummy = torch.ones(1, 64, dtype=torch.long)   # batch_size=1, seq_len=64
traced = torch.jit.trace(model, dummy)
traced.save('bilstm_traced.pt')
```

> **Avantage TorchScript** : le fichier `.pt` peut être chargé et exécuté en C++, sans Python installé.

### Partie 2 — Export ONNX (pattern TP10)

ONNX (Open Neural Network Exchange) est un format universel compatible avec :
- ONNX Runtime (Microsoft) — serveurs Linux/Windows
- TensorRT (NVIDIA) — GPU haute performance
- OpenVINO (Intel) — edge computing
- CoreML (Apple) — iOS/macOS

```python
torch.onnx.export(
    model, dummy, 'bilstm.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},  # dimensions variables
        'logits':    {0: 'batch_size'}
    }
)
```

`dynamic_axes` est essentiel : sans ça, le modèle n'accepterait qu'un `batch_size` et `seq_len` fixés.

### Partie 3 — Quantization dynamique (pattern TP10)

Réduit la précision des poids de `float32` (4 octets) à `int8` (1 octet).
Résultat attendu : **~4× plus léger**, inférence plus rapide sur CPU, perte d'accuracy < 0.1%.

```python
model_q = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},      # couches Linear : embedding proj + FC classifier
    dtype=torch.qint8
)
```

> Le BiLSTM contient des `nn.Linear` dans la couche FC et dans les projections LSTM internes.
> La quantization dynamique cible ces couches.

### Partie 4 — Benchmark 3 formats (pattern TP10)

| Format | Taille | Latence (1 tweet, CPU) | Accuracy |
|---|---|---|---|
| PyTorch (base) | ~3 Mo | ~2 ms | 98.08% |
| TorchScript | ~3 Mo | ~2 ms | identique |
| Quantizé (int8) | ~1 Mo | ~0.8 ms | ~98.07% |

**Mesure de latence** : 100 répétitions sur une seule séquence, après 10 warmup.

```python
def mesure_latence(model, input_tensor, n=100):
    for _ in range(10): model(input_tensor)     # warmup
    t0 = time.time()
    for _ in range(n): model(input_tensor)
    return (time.time() - t0) / n * 1000        # ms
```

### Partie 5 — Fonction d'inférence sur texte brut (pattern TP10)

Pipeline identique à `01_clustering.ipynb` :

```
Texte brut
    ↓  clean_tweet()    ← même fonction que Bloc 5 du notebook 01
    ↓  tokenize()       ← vocab custom, MAX_LEN=64
    ↓  pad_sequence()
    ↓  BiLSTM forward
    ↓  softmax → top-k
→ [(label, confiance), ...]
```

La cohérence avec le preprocessing d'entraînement est **critique** : utiliser un preprocessing différent
en production dégraderait les performances même si le modèle est parfait.

**Exemples de comportement** :

| Tweet | Prédit | Confiance |
|---|---|---|
| "my iphone keeps crashing" | problem_report | ~99% |
| "flight to NYC cancelled" | complaint | ~99% |
| "order not delivered" | follow_up | ~99% |
| "when will my order arrive?" | question | ~95% |

### Partie 6 — Mini API FastAPI — Mini API FastAPI (pattern TP10)

```
POST /predict  →  {"intention": "problem_report", "confiance": 0.99, "top3": [...]}
GET  /health   →  {"status": "ok", "modele": "BiLSTM"}
```

```python
@app.post("/predict")
def predict(payload: PredictionInput):
    cleaned   = clean_tweet(payload.text)
    token_ids = tokenize(cleaned)
    tensor    = pad_sequence([torch.tensor(token_ids)], batch_first=True)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_val, top_idx = torch.topk(probs, k=3)
    return {"intention": label_names[top_idx[0]], "confiance": top_val[0].item(), ...}
```

**Lancement** :
```bash
pip install fastapi uvicorn
uvicorn api_bilstm:app --reload --port 8000
```

**Test** :
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "my flight got cancelled and no one answered"}'
```

---

## Comparaison BiLSTM vs DistilBERT pour le déploiement

| Critère | BiLSTM | DistilBERT |
|---|---|---|
| Taille modèle | ~3 Mo | ~250 Mo |
| Latence (1 tweet, CPU) | ~2 ms | ~50 ms |
| Accuracy test set | 97.87% | **99.50%** |
| Dépendances | PyTorch seul | transformers + tokenizers |
| Déploiement mobile | ✅ facile | ⚠️ lourd |
| Interprétabilité | Moyenne | Faible |

**Conclusion** : DistilBERT est le modèle le plus performant avec 99.50% de test accuracy, soit ~2% de mieux que le BiLSTM (97.87%). Cet écart est significatif et s'explique par le pré-entraînement sur 11 milliards de tokens, qui donne à DistilBERT une compréhension contextuelle inaccessible aux modèles entraînés from scratch sur ~465 000 tweets.

Malgré cela, le **BiLSTM est retenu pour le déploiement** : il est 80× plus léger (~3 Mo vs ~250 Mo), 25× plus rapide (~2 ms vs ~50 ms par tweet), et ne requiert aucune dépendance HuggingFace en production. Dans un contexte de support client où des milliers de messages sont traités en temps réel, la latence et la portabilité priment sur un gain de 2% d'accuracy. Le BiLSTM quantizé (int8) réduit encore la taille à ~1 Mo pour ~0.8 ms de latence, sans perte mesurable de performance.

---

## Justification de l'approche par règles lexicales

### Pourquoi pas le clustering KMeans ?

L'exploration KMeans (notebook 01, K=4 à 14) a montré que les tweets de service client **ne sont pas géométriquement séparables** dans l'espace d'embedding. Tous les clusters tournent autour des mêmes mots génériques (`get`, `please`, `help`, `service`) quel que soit K. La raison : les tweets de SAV partagent un registre émotionnel commun qui efface les frontières sémantiques dans l'espace vectoriel.

Les règles lexicales sont donc **la seule alternative pratique** en l'absence d'annotations humaines.

### Plus de règles = meilleure couverture

C'est vrai : ajouter des mots-clés réduit le bucket `general`. Par exemple enrichir
la règle `follow_up` avec `"no news"`, `"waiting since"`, `"heard nothing"` récupère
des tweets qui tombaient dans `general`. C'est une démarche itérative : analyser les
exemples mal classifiés → identifier les mots manquants → enrichir les règles.

### Limite intrinsèque : les tweets ambigus

Même avec des règles exhaustives, certains tweets resteront dans `general` :

```
"I've been waiting for 3 days and nothing"     → follow_up (règle) ou general (si absent)
"absolutely terrible experience"               → complaint (règle) ou general (si absent)
"still no response from your team"             → follow_up via "still no response"
```

Ces tweets sont **genuinement ambigus** sans fil de conversation ou contexte supplémentaire.
`general` est la classe correcte pour ce qui reste non classifiable.

### Cadre théorique : supervision faible (Weak Supervision)

Cette approche correspond au paradigme de **supervision programmatique** (Snorkel, 2017) :
- Des fonctions de labellisation (`labeling functions`) définissent des règles heuristiques
- Elles sont rapides à créer, transparentes et contrôlables
- Le modèle apprend à **généraliser au-delà des règles exactes** : le BiLSTM peut classifier
  `"my device won't turn on"` comme `problem_report` si des patterns similaires apparaissent
  dans les données d'entraînement, même sans séquence exacte comme "not working"

C'est précisément ce qui justifie l'entraînement d'un modèle plutôt que d'utiliser
les règles directement en production : le modèle capture les **patterns contextuels**
autour des mots-clés, pas seulement les mots-clés eux-mêmes.

### Comportement en production

| Tweet | Règle directe | BiLSTM prédit | Justification |
|---|---|---|---|
| "my iphone is broken" | problem_report ✅ | problem_report | séquence "broken" présente |
| "my device won't start" | general | problem_report (probable) | pattern appris |
| "stuck at the airport" | general | complaint (probable) | frustration détectée |
| "I'm frustrated" | general | complaint | intensité émotionnelle apprise |

Le modèle est donc **plus robuste que les règles seules** — c'est la valeur ajoutée de l'apprentissage.

---

## Exemples réels — API en production

Tests effectués sur l'API déployée (`http://51.15.234.209:8000/predict`) :

**Exemple 1 — Cas ambigu → general**
```json
{"text": "@AppleSupport fix your fucking health app. I just went for a 1 hour 20 minutes jog."}
```
```json
{"intention": "general", "confiance": 0.7909, "top3": ["general 0.79", "problem_report 0.18", "help_request 0.02"]}
```
> La mention `@AppleSupport` est supprimée par le preprocessing. "fix your app" n'active aucun mot-clé exact (`not working`, `crash`...). Confidence faible (0.79) avec `problem_report` en 2e position à 0.18 — le modèle hésite, ce qui est cohérent : la phrase est une plainte implicite sans signal fort.

**Exemple 2 — Question claire → question**
```json
{"text": "Does it always seem to get hot when on phone calls, or was this the first time? When did the battery issues begin?"}
```
```json
{"intention": "question", "confiance": 0.9996, "top3": ["question 0.9996", "general 0.0003", "problem_report 0.0"]}
```
> Deux `?` + marqueurs interrogatifs (`when`, `or`) → classification `question` quasi-certaine (0.9996). Cas idéal : forme et fond alignés.

**Exemple 3 — Demande agent → general**
```json
{"text": "Can you send your full name, postcode & address and email please? Thanks"}
```
```json
{"intention": "general", "confiance": 0.9827, "top3": ["general 0.9827", "help_request 0.0086", "problem_report 0.0036"]}
```
> Tweet d'un agent de support (non-client) demandant des informations. "Can you send" ne correspond à aucune règle `help_request` ("please help", "need help"). Classé `general` correctement — le modèle n'est pas trompé par la structure interrogative.

**Exemple 4 — Sarcasme → limite du modèle**
```json
{"text": "@comcastcares I'm paying $80 a month instead of $50 because your promotion is 'over'. And now to top that off your service goes out at prime Netflix time. Thanks!❤️"}
```
```json
{"intention": "general", "confiance": 0.9959, "top3": ["general 0.9959", "problem_report 0.0031", "help_request 0.0004"]}
```
> Sémantiquement c'est une plainte évidente — client sarcastique ("Thanks!❤️") qui cumule une augmentation de prix et une coupure de service. Mais aucun mot-clé explicite de `complaint` (`terrible`, `awful`, `unacceptable`...) n'est présent. Pendant l'entraînement, ce tweet a donc été labelisé `general` par les règles. Le modèle reproduit fidèlement ce label à 99.59%.
>
> **Limite fondamentale** : le modèle ne peut pas apprendre ce que les règles n'ont pas su capturer. Le sarcasme, l'ironie et les plaintes implicites contournent systématiquement les règles lexicales et tombent en `general`. Pour détecter ces cas, il faudrait soit annoter manuellement des exemples sarcastiques, soit utiliser un modèle pré-entraîné sur de la détection de sentiment/sarcasme.

---

## Pistes d'amélioration

### 1. Weak Supervision enrichie (Snorkel)

L'approche actuelle utilise une seule fonction de labellisation (règles lexicales). Une amélioration naturelle serait d'en combiner plusieurs via le framework **Snorkel** :

- **Règles lexicales** (actuelles) : mots-clés d'intention
- **Modèle de sentiment** : utiliser `cardiffnlp/twitter-roberta-base-sentiment` pour détecter les plaintes implicites et le sarcasme — un tweet avec sentiment très négatif sans mot-clé explicite serait labelisé `complaint`
- **Patterns syntaxiques** : détection de structures sarcastiques ("Thanks!❤️", "great job...", guillemets ironiques)

Les fonctions de labellisation sont combinées par un modèle génératif qui résout les conflits — le label final est plus robuste qu'une règle seule.

### 2. Classification zero-shot

Utiliser directement un modèle pré-entraîné pour classifier sans fine-tuning :

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(tweet, candidate_labels=["complaint", "question", "problem_report", ...])
```

**Avantage** : aucune donnée labelisée requise, comprend le sarcasme et le contexte implicite nativement.

**Inconvénient** : lent (~500 ms par tweet vs ~2 ms pour le BiLSTM), coûteux à l'échelle de millions de tweets, moins contrôlable. Non adapté à un système de support client temps réel.

### Conclusion

Le système actuel est solide pour les intentions explicites (97-99% accuracy sur les classes bien couvertes par les règles). Les limites apparaissent sur le sarcasme et les expressions implicites — cas qui nécessitent une compréhension pragmatique du langage que les labels keyword-based ne peuvent pas encoder.

> **Nuance importante** : une accuracy élevée ne signifie pas que le modèle est fiable à 100%. L'accuracy mesurée (98.08%) reflète la capacité du modèle à reproduire les labels générés par les règles — pas sa capacité à comprendre l'intention réelle d'un tweet. Un tweet sarcastique sera classé `general` avec 99% de confiance, ce qui est "correct" vis-à-vis des labels d'entraînement, mais faux vis-à-vis de l'intention humaine réelle. La métrique d'accuracy est donc à interpréter dans le contexte de la qualité des labels, pas comme une garantie absolue de performance en production.