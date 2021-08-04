# URL Multi-Label Classification

  
  

## Installation

```sh

pip install requirements.txt

```
- Lancer `main.py` après avoir choisi son modèle.
- Lancer jupyter pour les notebooks :
    - PART1_Exploratory_data_analysis: Cette partie nous permet d'extraire les variables importantes, identifier les valeurs aberrantes et manquantes et ainsi nettoyer l'ensemble de nos données. Maximiser nos informations sur la colonne `URL` et mieux comprendre sa relation avec les  autres variables.
      - PART2_Exploratory_data_analysis: Cette partie est la suite de la partie une et effectue une analyse exploratoire sur la variable `jour`.
    -  PART 3 et PART 4 et PART 5: entrainements des modèles  

## Récapitulatif du travail effectué :

### Preprocessing des URLs

#### Pré-traitement :

Cette étape permet de standardiser le texte afin de rendre son usage plus facile.
- Normalisation: minisucule, majuscule.
- Tokenization : C’est probablement l’étape la plus importante et la plus complexe du processus et représente un point majeur dans l'obtention des résultats. La tokenisation cherche à transformer un texte en une série de tokens individuels. En général nous avons :
* Tokenizers basées sur les règles, comme spacy par exemple. Cependant ce type ne gère pas efficacement les mots rares.

* Tokenisation en sous-mots : comme byte-pair encoding (BPE), wordpiece, unigram language model, sentencepiece. Ce deuxième type peut gerer les mots rares et inconnes. Néaumoins, les tokenizers de sous-mots sont appris à partir de données, la qualité et la quantité des données sont cruciales pour obtenir de bonnes performances. La construction de vocabulaire peut soit se faire traditionnellement ou avec hachage . Lorsque nous n'avons pas beaucoup de données, il peut être préférable d’utiliser un tokenizer basé sur des règles. Nous pouvons dans ce cas là, utiliser des tockenizers qui sont déjà construits sans avoir à en construire un.



**A noter:** Moins de normalisation et de ségmentation tend à conduire à un vocabulaire plus important. Par conséquant, une taille de vocabulaire plus importante peut ralentir l’entraînement et peut aussi causer des problèmes de mémoire.



**Numérisation (Représentation du texte en format numérique) :** 

Cette étape peut être effectuée via des techniques de sac de mots (Bag of Words) ou Term Frequency-Inverse Document Frequency (Tf-IdF). On peut également apprendre des représentations vectorielles (embedding) par apprentissage profond. BOW: Le nombre de mots n'est pas toujours la meilleure idée car il y a des mots dans les URL qui sont plus importants que les autres mots.

  

#### Inconvénient du Bag of Words:
- Ignore l'ordre des mots dans notre phrase
- N'a pas une compréhension sémantique
- Taille du vecteur trop importante et contient beaucoup de valeurs nulles.

Nous pourrons commencer par utiliser TF-IDF.

  **Feature Engineering**

Nous créerons des colonnes supplémentaires à partir de la colonne `URL`.

  

**Le stemming et la lemmatization**

Ce sont des formes plus poussées de normalisation, comme extraire la racine du mot(stemming) ou enlever les prèfixes des mots (lemmatization) ou unifier les mots qui se ressemblent. Dans notre cas ou nous avons des URLs.

  Nous allons combiner plusieurs de ces techniques présentées puis résumerons les résultats dans un tableau tout à la fin

* Nous testerons différentes approches afin de tokeniser les URLS car ils ne ressemblent pas à un texte de document commun.

* A Première vue, les mots des liens sont séparés par des: tirets, Majuscules, +, ",", !, : , _ , %, et peut etre d'autres caractères spéciaux: # ; @ ? & $ ~ crochets et accolades . Il semble aussi y avoir des mots collés comme par exemple : "TennisFicheJoueur1500000000003017.html"

* Nous pouvons également au début essayer de séparer les sites, ainsi que les mots des chemins des URLs. Nous pouvons séparer les mots collés qui commencent par des majuscules.

L'URL est en général composé d'un : domaine, sous-domaine, protocole, chemin du fichier et la requête. Dans un premier nous pourrons créer des catégories à partir de l'url et voir ce que ça donne.
 
 Quelques statistiques ont montré que notre dataset est composé majoritairement du site `cdiscount` avec plus de 10000 exemples. Il existe aussi des domaines avec une seule occurrence !


#### Aller plus loin dans le feature engineering

Nous pouvons aller encore plus loin avec le feature engineering, comme par exemple: combiner le sous_domaine avec le path / essayer de comprendre les relations entre le contenu du path et les différentes colonnes de notre dataset, , de créer des sous catégories à partir de path pour les mêmes sites.

Pour cette première version de notre Exploratory Data Analysis, nous allons nous arreter là et proceder à la tokenisation du path afin de créer une description à partir de ce dernier.

  

Pour la suite, nous pouvons utiliser et combiner plusieurs des techniques suivantes sur colonne path_description :

  

- Les "stop words" sont des mots qui se retrouvent très fréquemment dans la langue française. Ces mots n’apportent pas d’information dans les tâches de NLP. Aussi, nos URLS ne contiennent pas de mots avec des accents et il serait donc judicieux d'enlever les accents des stop words

- Enlever les mots htm, php, aspx, html ? test avec et sans et voir la corrélation

- Supprimer meme mots qui se suient (chambre-literie/literie-sommier)

- Supprimez tous les caractères non pertinents tels que les caractères non alphanumériques et numéro bizarres et longs

- Convertir les mots en minuscule.

- Essayer de combiner les mots mal orthographiés ou orthographiés alternativement en une seule représentation, séparer les mots collés en minuscule (sauter cette étape je pense)

- Lemmatization et stemming

- Enlever les mots composés de 1 seul caractère

- Enlever les chiffres (à voir)

- Enlever les mots spéciaux: htm, php, aspx, html (peut afficher le dernier token puis faire un count pour les retrouver)

- Utiliser les tokenizers simples ou utiliser les tokenizers déja pré-entrainés comme celui de bert qui inclus : BPE, (lowercase et Uppercase), pour le transformer



Nous utilisons une classe que nous avons créer permettant de tokenizer les URLs deu dataframe en combinant plusieurs des techniques présentées en haut. Elle pourrait etre améliorer plus tard (performance et facilité d'utilisation) dépendamment du temps que j'aurai.


### Multi-Label Analysis :

Contrairement à la classification où chaque exemple est associé une seule classe, la classification multi-label associe plusieurs catégories à un exemple.

**Severe umbalanced dataset** : 
La distribution des classes est inégale (par exemple 1:100 ou plus).
Le nombre de targets n'est pas le meme dans le dataset, Nous avons des targets qui apparaissent plus de 3000 fois tandis que d'autres apparaissent des centaines de fois ou qu'une seule fois.

**Méthodes pour gérer ce cas:**
- Over Sampling et Undersampling (augmenter la fréquence de la classe minoritaire ou diminuer la fréquence de la classe majoritaire)
- Ensemble Techniques: bagging / boosting 

Nous avons en tout 67595 exemples avec 1903 targets. Il y a jusqu'à 59060 d'exemples qui contiennent 5 targets, 4951 qui contiennent 4 targets, 1252 contiennent 3 targets (également, 1200 et 1000 pour 2 et 1 target)
  

### Stratégies essayées :
** j'ai voulu maximiser les moments où je lançais les entrainements pour exploiter ce temps afin d'apprendre en parallèle d'autres stratégies et méthodes pour attaquer ce problème de multi-label classification et en apprendre plus à ce sujet. **

###  Réduction de la dimension des labels 
Nous avons 1903 labels pour notre target ce qui reprensente un nombre assez elevé pour de la classification multi-label.
Nous allons donc effectuer une réduction sur l'espace de dimension des labels.
Plusieurs algorithmes existent pour la réduction de l'espace des labels: 
- Compressed Sensing (CS) et la  Principal Label Space Transformation (PLST) qui est l'équivalente du PCA sur les features. Egalement des techniques d'embeddings entre targets
- Contrairement à ces techniques qui sont indépendentes de nos entrées, il en existe aussi d'autres qui le sont comme par exemple la CPLST ( Conditional Principal Label Space Transformation ) 

- Un autre moyen serait d'essayer d'utiliser une approches par embedding sur les labels afin d'extraire les labels qui sont souvent ensemble
- Ou utiliser DEFRAG pour ce genre de problème ( Extreme multilabel classification)
- 
Pour la classification multi-label, il existe une librarie nommée `scikit-multilearn` qui s'appuie sur la librarie scikit-learn. Elle contient aussi un wrapper autour de  MEKA qui propose une implémentation des méthodes d'apprentissage et d'évaluation multi-labels.

#### PLST :
PLST or Principal Label Space Transformation est une méthode de réduction de dimensionnalité utilisée explicitement pour les labels des datasets et permettant de les représenter géométriquement tout en minimisant l'erreur de reconstruction, nous combinons cette méthode ensuite avec des baselines modèles pour l'apprentissage.
  

#### Label Graph :
C'est une méthode permettant de représenter un embedding des différents labels de l'output en créant ainsi un graphe de lien représentations des interactions continues entre les différents labels attendues. C'est un modèle qui a démontré dans la littérature une accuracy élevé et des résultats très concluants dans le domaine de la classification multi-label.


### Création des modèles :
#### Baseslines: 
- Dans un premier temps, nous allons construire nos modèles Baselines avec l'ensemble des targets présentes dans notre dataset. Chaque modèle essaiera de prédire les targets sur soit le nombre total de 1009 targets ( Ce qui n'est peut etre pas une bonne approche), ou sur la réduction de dimension des 1009 targets. 
- Après avoir construit les baslines models, nous essaierons d'améliorer les modèles avec des hyperparamètres pour voir s'il y a une augmentation du score F1.


#### 1.  OneVsRest: 
Le problème est décomposé en un problème de classification binaire multiple. Nous choisissons une classe et formons un classificateur binaire avec les échantillons de la classe sélectionnée d'un côté et tous les autres échantillons de l'autre côté. Ainsi, nous obtiendrons N classificateurs pour N étiquettes et lors du test nous classerons simplement l'échantillon comme appartenant à la classe avec le score maximum parmi les N classificateurs.
 
  
#### Multi-Label Binary Relevance :
Transforme un problème de classification multi-étiquettes avec L étiquettes en L problèmes de classification binaire séparés à une seule étiquette en utilisant le même classificateur de base fourni dans le constructeur. La sortie de prédiction est l'union de tous les classifieurs par étiquette. S'en suit ensuite l'apprentissage d'un modèle multinomial naive bayes.