# 01 — Exploration et clustering

## Objectif

Le dataset Twitter contient ~3 millions de tweets non labellisés. Pour entraîner un modèle de classification supervisé, il faut des paires (tweet, intention). Ce notebook produit ces paires automatiquement :

1. Nettoyer les données brutes **en préservant les marqueurs d'intention**
2. Découvrir les groupes naturels via KMeans
3. Lire les mots-clés de chaque groupe → nommer les classes

**Principe fondamental** : on ne définit pas les classes à l'avance. Les données
révèlent leur propre structure, on la lit, puis on nomme ce qu'on a trouvé.

On traite uniquement les tweets clients (`inbound=True`). Les tweets entreprise ne sont pas utilisés — l'objectif est de classifier les messages entrants, pas les réponses.

**Classification par intention, pas par domaine** : un tweet "Mon iPhone est cassé" et
"Mon colis n'est pas arrivé" ont la même intention (`problem_report`) malgré des sujets
différents. On cherche la fonction communicative, pas le topic.

---

## Structure du notebook

### Bloc 1 — Imports
Uniquement les imports. Rien d'autre dans ce bloc.

### Bloc 2 — Chargement CSV
On charge `twcs.csv` et on isole `df_client` (`inbound=True`).

### Bloc 3 — Diagnostic avant nettoyage
Avant de supprimer quoi que ce soit, on affiche ce qui serait supprimé : NaN dans `text`, NaN dans `tweet_id`, doublons sur `tweet_id`, tweets vides ou trop courts. Ce bloc ne modifie rien.

### Bloc 4 — Nettoyage structurel
On supprime uniquement les tweets de moins de 3 caractères.
Les lignes `dropna` et `drop_duplicates` sont volontairement commentées : le diagnostic Bloc 3 montre que les NaN et doublons sont inexistants sur ce dataset.

### Bloc 5 — Nettoyage textuel orienté intention

On applique `clean_tweet_intent` : supprime `@mentions`, URLs, hashtags, chiffres, et
la ponctuation **sauf `?`, `!`, `'` (apostrophe)**.

```python
def clean_tweet_intent(text):
    text = re.sub(r'@\w+', '', text)               # @mentions → bruit de marque
    text = re.sub(r'http\S+', '', text)             # URLs
    text = re.sub(r'#\w+', '', text)                # hashtags
    text = re.sub(r'\d+', '', text)                 # chiffres
    text = re.sub(r"[^\w\s?!']", ' ', text)         # ponctuation sauf ? ! '
    text = re.sub(r'\s+', ' ', text)                # espaces multiples
    return text.strip().lower()
```

**Pourquoi conserver `?` et `!`** :

| Marqueur | Signal d'intention |
|---|---|
| `?` | Demande d'information — distingue `question` des autres classes |
| `!` | Intensité émotionnelle — renforce `complaint` ou `positive_feedback` |
| `'` (apostrophe) | Préserve `can't`, `don't`, `won't` — marqueurs de négation critiques |

**Pourquoi supprimer les `@mentions`** : les mentions (`@AppleSupport`, `@AmazonHelp`)
sont du bruit de domaine. Leur présence pousse KMeans à créer des clusters Apple, Amazon,
Uber — pas des clusters d'intention.

Les stopwords sont conservés pour l'embedding et filtrés uniquement à l'affichage des mots-clés.

### Bloc 6 — Configuration et embeddings

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| `SAMPLE_CLUSTER` | 200 000 | Échantillon représentatif faisable en ~3 min sur P100. |
| `K_MIN` | 4 | En dessous, les clusters sont trop larges. |
| `K_MAX` | 14 | Au-delà, les clusters se dupliquent ou deviennent trop spécifiques. |
| `RANDOM_STATE` | 42 | Reproductibilité. |
| `BATCH_SIZE` | 256 | Adapté à la VRAM 16 GB du P100 pour roberta-base (768d). |

On encode `text_clean` avec `cardiffnlp/twitter-roberta-base`, pré-entraîné sur 58 millions de tweets.

### Bloc 7 — Exploration KMeans (K=4 à 14)

Pour chaque K, on affiche par cluster : nombre de tweets, 5 mots-clés, et trois signaux
d'intention : `?%`, `!%`, longueur moyenne.

```python
STOP_INTENT = STOP - {'not', 'no', 'never', 'please', 'still', 'when', 'how',
                      'what', 'why', 'where', 'help', 'cant', 'wont', 'dont'}
```

`STOP_INTENT` retire les stopwords génériques mais conserve les marqueurs d'intention
(`not`, `please`, `when`...) pour qu'ils apparaissent dans les top5 des clusters.

**Résultat de l'exploration** : quatre clusters sont géométriquement séparables quel que soit K :

| Cluster | Mots-clés stables | Signal | Apparaît à partir de |
|---|---|---|---|
| `acknowledgment` | thanks, sent, done, yes | `? 5-7%`, len 1-3 | K=4 |
| `non_english` | de, que, la, no, en | `? 21%`, langue romane | K=4 |
| `non_english` (asiatique) | क, ह, म, र, न | `? 2%`, scripts non-latins | K=8 |
| `question` | how, when, what, why | `? 40-53%` | K=9 |

À partir de K=8, deux clusters supplémentaires émergent partiellement :

| Cluster | Mots-clés | Signal |
|---|---|---|
| `problem_report` | not, phone, iphone, app | `? 30-35%` |
| `complaint` (voyage) | flight, not, no, service | `! 22-24%` |

**Pourquoi KMeans échoue pour l'intention** : les embeddings capturent la similarité
sémantique entre "mon iPhone est cassé" et "mon colis n'arrive pas" — deux tweets avec
une intention identique (`problem_report`) mais des topics différents. Les mots de contenu
(iPhone, colis) éloignent les points autant que les mots d'intention les rapprochent.

**Conclusion** : KMeans ne peut pas découvrir des classes d'intention sur ce dataset.
On abandonne les IDs de clusters comme labels et on passe à des règles explicites
basées sur les marqueurs linguistiques d'intention.

---

### Bloc 8 — Labellisation par intention

Au lieu d'utiliser les IDs de clusters KMeans, on définit des règles sur les
**marqueurs d'intention** : ponctuation (`?`, `!`), verbes, mots de sentiment,
structure de la phrase.

**Taxonomie des 8 intentions** :

| Classe | Fonction communicative | Exemple |
|---|---|---|
| `non_english` | Tweets non-anglais (filtrage) | "Hola @AmazonHelp me pueden..." |
| `acknowledgment` | Confirmer, remercier brièvement | "Thanks, sent!" |
| `question` | Demander une information | "When will my order arrive?" |
| `problem_report` | Décrire un dysfonctionnement | "App keeps crashing since update" |
| `complaint` | Exprimer frustration / insatisfaction | "This is absolutely unacceptable!!" |
| `follow_up` | Relancer une interaction précédente | "Still waiting, any update?" |
| `help_request` | Demander une action | "Please fix this, I need help urgently" |
| `general` | Catch-all | Tout le reste |

**Implémentation** :

```python
NON_EN = {'que','pas','por','con','pour','vous','para','de','la','en','les','du',
          'ich','die','der','das','und','ist',
          'per','non','una','sono','che',
          'est','une','sur','avec','mais','tout',
          'yang','untuk','dengan','tidak','saya'}

def assign_label(text):
    words = set(text.lower().split())
    text_lower = text.lower()

    if sum(1 for c in text if ord(c) > 0x3000) > 3:   # scripts asiatiques
        return 'non_english'
    if len(words & NON_EN) >= 2:
        return 'non_english'

    if len(words) <= 10 and words & {'thanks','thank','sent','done','yes','ok','noted','received'}:
        return 'acknowledgment'

    if '?' in text and words & {'when','how','what','where','why','which'}:
        return 'question'

    problem_seq = ['not working',"doesn't work",'broken','error','crash',
                   'frozen','bug','glitch','failed','unable','not loading']
    if any(s in text_lower for s in problem_seq):
        return 'problem_report'

    if words & {'worst','terrible','awful','unacceptable','horrible',
                'ridiculous','shame','disappointed','useless','rude','disgrace'}:
        return 'complaint'

    follow_seq = ['still waiting','any update','still no','no response','been waiting']
    if any(s in text_lower for s in follow_seq):
        return 'follow_up'

    help_seq = ['please help','help me','need help','can you help','please fix','please check']
    if any(s in text_lower for s in help_seq):
        return 'help_request'

    return 'general'
```

Avant d'appliquer les règles, on filtre les tweets devenus vides après nettoyage :

```python
df_client = df_client[df_client['text_clean'].str.strip().str.len() >= 3].reset_index(drop=True)
df_client['label'] = df_client['text_clean'].apply(assign_label)
```

**Distribution obtenue sur 1 511 278 tweets clients** :

| Classe | Tweets | % |
|---|---|---|
| general | 1 068 237 | 70.7% |
| question | 178 099 | 11.8% |
| acknowledgment | 74 617 | 4.9% |
| problem_report | 54 502 | 3.6% |
| complaint | 48 708 | 3.2% |
| non_english | 36 444 | 2.4% |
| help_request | 30 718 | 2.0% |
| follow_up | 19 953 | 1.3% |

**70% de `general` n'est pas un problème** : avec `N_MAX=100K` par classe, le dataset
d'entraînement sera équilibré (~465K tweets, 8 classes). Le modèle voit autant
d'exemples de `general` que de `question`.

**Limite connue — labels déterministes et accuracy élevée**

Les labels produits par `assign_label` sont **entièrement déterministes** : un tweet identique produira toujours le même label. Conséquence directe : les modèles entraînés sur ces données convergent à des accuracies très élevées (>99%), car ils apprennent à reproduire des règles exactes plutôt qu'à résoudre une ambiguïté réelle de classification.

Ce résultat est attendu et constitue une limite connue de l'approche, pas une anomalie :

- **Pourquoi ne pas annoter manuellement ?** Le dataset contient 1,5 million de tweets clients. Une annotation humaine est hors de portée — les règles lexicales sont la seule alternative pratique (voir aussi la section "Weak Supervision" dans `04_deploiement.md`).
- **Pourquoi 8 classes ?** C'est ce que l'exploration KMeans (K=4 à 14) a révélé comme structurellement distinguable dans les données. Les 4 clusters stables (acknowledgment, non\_english, question, et le reste) plus 2 patterns émergents à K=8 (problem\_report, complaint) plus 2 intentions lexicalement identifiables (follow\_up, help\_request), et un catch-all `general` pour tout ce qui reste ambigu.
- **Valeur malgré tout** : le modèle généralise *au-delà* des règles exactes. Un tweet comme `"my device won't start"` peut être prédit `problem_report` sans correspondre à une séquence dans les règles — c'est le pattern contextuel appris sur les données d'entraînement.

---

### Bloc 8b — Re-clustering du général

Pour explorer si des sous-intentions sont cachées dans `general`, on re-encode
uniquement ces tweets et on relance KMeans (K=4 à 10).

```python
df_general = df_client[df_client['label'] == 'general'].copy()
sample_gen = df_general.sample(min(200_000, len(df_general)), random_state=RANDOM_STATE)
emb_gen = embed_model.encode(sample_gen['text_clean'].tolist(), ...)

for k in range(4, 11):
    show_clusters(sample_gen, emb_gen, k, 'GENERAL')
```

**Résultats** : le re-clustering confirme que `general` ne se décompose pas en intentions
claires. Deux corrections utiles ont été identifiées et intégrées dans la règle `assign_label` :

1. **Non-anglais résiduels** : ~6 300 tweets en allemand, italien, malais non capturés —
   ajout de `ich`, `die`, `der`, `per`, `yang`... dans `NON_EN`.

2. **Tweets vides après nettoyage** : ~26 500 tweets réduits à une URL seule —
   filtrés par `str.len() >= 3` avant labellisation.

---

### Bloc 9 — Validation des labels
Pour chaque classe : affichage de 5 tweets exemples + % de couverture.
Permet de vérifier visuellement que les règles capturent les bons tweets.

---

### Bloc 10 — Équilibrage
`general` représente 70.7% du dataset. Sans équilibrage, le modèle ignorerait les classes minoritaires.

On plafonne chaque classe à `N_MAX = 100 000` exemples :
```python
df_bal = df_client.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), N_MAX)))
```
Les classes rares (`follow_up`, `help_request`, `non_english`) conservent tous leurs exemples.
`general` et `question` sont sous-échantillonnés à 100K.

### Bloc 11 — Split stratifié 70/15/15
On divise `df_bal` en trois ensembles :

| Ensemble | % | Usage |
|---|---|---|
| Train | 70% | Entraînement du modèle |
| Val | 15% | Suivi de la loss, early stopping |
| Test | 15% | Évaluation finale (touché une seule fois) |

Le paramètre `stratify=y` garantit que chaque split respecte les proportions de classes.

### Bloc 12 — Vocabulaire
Construit à partir du train uniquement (évite le data leakage) :
- On compte les occurrences de chaque token dans `X_tr`
- On garde les `VOCAB_SIZE = 15 000` tokens les plus fréquents
- Deux tokens spéciaux : `<PAD>` (id=0) pour le padding, `<UNK>` (id=1) pour les tokens hors-vocabulaire

### Bloc 13 — Label mapping + Tokenisation
Ce bloc fait deux choses en une passe :

**Label mapping** : les labels string sont convertis en entiers.
```python
LABEL_NAMES = sorted(df_bal['label'].unique())   # ordre alphabétique → reproductible
label2id    = {l: i for i, l in enumerate(LABEL_NAMES)}
id2label    = {i: l for l, i in label2id.items()}
```

Ordre alphabétique des 8 classes :
```
0: acknowledgment   1: complaint    2: follow_up    3: general
4: help_request     5: non_english  6: problem_report  7: question
```

**Tokenisation** : chaque tweet est converti en liste d'IDs de tokens, tronquée à `MAX_LEN = 64`.
```python
def tokenize(text):
    tokens = text.split()[:MAX_LEN]
    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    return ids if ids else [vocab['<UNK>']]
```
Les trois datasets sont produits en une seule compréhension : `(token_ids, label_int)`.

### Bloc 14 — Sauvegarde
Quatre fichiers persistés dans `../data/` :

| Fichier | Taille | Contenu | Utilisé par |
|---|---|---|---|
| `vocab.pkl` | 0.2 MB | `{token: id}` (15 000 entrées) | CNN, BiLSTM, Transformer |
| `label_maps.json` | ~0 MB | `{id: label}` pour reconstruction | Tous les modèles |
| `client_data.pkl` | 22.3 MB | `{train/val/test: [(token_ids, label_int)]}` | CNN, BiLSTM, Transformer |
| `client_text_data.pkl` | 45.4 MB | `{train/val/test: [(text_clean, label_int)]}` | DistilBERT (tokenise lui-même) |

### Bloc 15 — Vérification
Affiche la taille de chaque fichier en MB pour confirmer que la sauvegarde est complète.