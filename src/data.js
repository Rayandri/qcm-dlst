
// --- DATA: FLASHCARDS (CONCEPTS CLÉS) ---
export const flashcards = [
  {
    id: 1,
    title: "Le Triangle de Stationnarité AR(2)",
    content: "Pour $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\epsilon_t$. \nLes racines du polynôme $(1 - \\phi_1 z - \\phi_2 z^2)$ doivent être hors du cercle unité.\nConditions strictes :\n1. $\\phi_1 + \\phi_2 < 1$\n2. $\\phi_2 - \\phi_1 < 1$\n3. $|\\phi_2| < 1$",
    icon: "triangle"
  },
  {
    id: 2,
    title: "Stationnarité (Weak Sense)",
    content: "Une série est faiblement stationnaire si :\n• Moyenne constante : $\\mathbb{E}[X_t] = \\mu$\n• Variance constante dans le temps\n• Autocovariance dépend uniquement du lag $h$, pas de $t$\nCondition souvent requise pour les modèles stats (ARIMA).",
    icon: "chart"
  },
  {
    id: 3,
    title: "Bruit Blanc (White Noise)",
    content: "Le hasard pur. Imprévisible.\n• Moyenne nulle : $\\mathbb{E}[w_t] = 0$\n• Variance constante : $\\sigma^2$\n• Aucune corrélation : $Cov(w_t, w_s) = 0$ si $t \\neq s$\nSi les résidus de ton modèle sont un bruit blanc, tu as tout extrait !",
    icon: "metric"
  },
  {
    id: 4,
    title: "LSTM Gates",
    content: "3 portes contrôlent le flux d'information :\n• **Forget Gate** : Ce qu'on jette de la mémoire\n• **Input Gate** : Ce qu'on écrit dans la mémoire\n• **Output Gate** : Ce qu'on lit de la mémoire\nFormule Cell State : $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$",
    icon: "network"
  },
  {
    id: 5,
    title: "Attention Mechanism",
    content: "Le cœur du Transformer. 'Quelle partie de l'entrée est importante pour ce que je génère maintenant ?'\n$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$\nQ=Query, K=Key, V=Value. Division par $\\sqrt{d_k}$ pour éviter saturation du softmax.",
    icon: "eye"
  },
  {
    id: 6,
    title: "Positional Encoding",
    content: "Le Transformer n'a pas de récurrence → pas de notion d'ordre.\nOn ajoute des vecteurs de position (sin/cos) aux embeddings.\nSans PE : 'Manger pour vivre' = 'Vivre pour manger'\nFormule : $PE_{pos,2i} = sin(pos/10000^{2i/d})$",
    icon: "triangle"
  },
  {
    id: 7,
    title: "Vanishing Gradient (RNN)",
    content: "En BPTT, le gradient est multiplié par $W_h$ à chaque pas de temps.\nSi $|\\lambda_{max}| < 1$, le gradient → 0 exponentiellement.\n$0.9^{100} \\approx 0$ : oubli du passé lointain !\nSolutions : LSTM/GRU, Skip connections, initialisation orthogonale.",
    icon: "network"
  },
  {
    id: 8,
    title: "Contrastive Loss",
    content: "Principe du Contrastive Learning :\n$\\downarrow$ distance(ancrage, positif)\n$\\uparrow$ distance(ancrage, négatif)\nPaire positive = augmentations du même exemple\nPaire négative = exemples différents du batch",
    icon: "mask"
  },
  {
    id: 9,
    title: "SSL : InfoNCE Loss",
    content: "Loss standard du Contrastive Learning.\n$L = -\\log \\frac{\\exp(sim(z_i, z_j)/\\tau)}{\\sum_{k} \\exp(sim(z_i, z_k)/\\tau)}$\nC'est une cross-entropy catégorielle où la classe correcte est la paire positive. $\\tau$ = température.",
    icon: "mask"
  },
  {
    id: 10,
    title: "Linear Probing",
    content: "Évaluation de la qualité des features SSL :\n1. Geler le backbone pré-entraîné\n2. Entraîner seulement un classifieur linéaire dessus\nSi ça marche bien → le modèle a 'compris' les données.\nAlternative : Fine-tuning (ré-entraîner tout).",
    icon: "chart"
  },
  {
    id: 11,
    title: "Dimensional Collapse",
    content: "Problème en SSL : le modèle 'triche' en sortant toujours le même vecteur.\nConséquence : loss faible mais représentations inutiles.\nSolutions :\n• Paires négatives (SimCLR)\n• Asymmetric networks (BYOL/SimSiam)\n• Régularisation VICReg",
    icon: "chart"
  },
  {
    id: 12,
    title: "Channel Independence (PatchTST)",
    content: "Innovation de PatchTST (SOTA 2023) :\nTraiter chaque variable d'une série multivariée comme une série univariée distincte.\n→ Partage des poids, moins de paramètres\n→ Meilleure généralisation que le 'Channel-Mixing'",
    icon: "triangle"
  }
];

// --- DATA: SLIDE SUMMARIES (CHEAT SHEETS) ---
export const slideSummaries = {
  session1: {
    title: "Session 1 : Concepts & Tâches",
    points: [
      { t: "Définition", c: "Série Temporelle : Suite d'observations indexées par le temps $X = \\{X_{t_1}, ..., X_{t_n}\\}$. Processus Stochastique : La loi de probabilité génératrice." },
      { t: "Stationnarité", c: "Faible : Espérance constante, Variance constante, Autocovariance ne dépend que du délai $h$ (pas du temps $t$)." },
      { t: "Modèles Stats", c: "Random Walk (Non stationnaire), AR(p) (Linéaire), Prophet (Additif : Tendance + Saison + Vacances), Kalman (État caché + Mesure)." },
      { t: "Métriques", c: "MSE (Sensible outliers), MAE (Robuste), MAPE (Pourcentage, attention si $y=0$). ACF/PACF pour l'analyse de dépendance." }
    ]
  },
  session2: {
    title: "Session 2 : Architectures DNN",
    points: [
      { t: "RNN vs LSTM/GRU", c: "RNN : Vanishing Gradient. LSTM : 3 gates (Input, Forget, Output). GRU : 2 gates (Reset, Update), pas de Cell State séparé." },
      { t: "CNN 1D (TCN)", c: "Convolutions causales (pas de futur) & dilatées (grand champ réceptif sans perte de résolution). Parallélisables." },
      { t: "Transformers", c: "Attention $O(L^2)$. Invariant par permutation $\\to$ Positional Encoding nécessaire. PatchTST : Channel Independence (SOTA)." },
      { t: "Régularisation", c: "Skip Connections (ResNet) pour la profondeur. Dropout. Layer Norm (préférée au Batch Norm pour les séquences)." }
    ]
  },
  session3: {
    title: "Session 3 : Self-Supervised Learning",
    points: [
      { t: "Le Concept", c: "Apprendre sans labels via une 'Tâche Prétexte' (Masking, Contrastive). Le 'Gâteau' de LeCun : SSL = la masse du gâteau." },
      { t: "Contrastive Learning", c: "Rapprocher $(x, x^+)$ (Positifs) et éloigner $(x, x^-)$ (Négatifs). Loss : InfoNCE. Augmentations : Jitter, Scale, Warping." },
      { t: "Problèmes SSL", c: "Dimensional Collapse (Triche par vecteur constant). Solutions : Paires négatives (SimCLR) ou Asymétrie (BYOL)." },
      { t: "Evaluation", c: "Linear Probing (Backbone gelé + Classifieur linéaire) pour tester la qualité des embeddings. Fine-tuning pour la tâche finale." }
    ]
  },
  session4: {
    title: "Session 4 : Avancé / SOTA",
    points: [
      { t: "Forecasting Stratégies", c: "Recursive (accumule erreurs) vs Direct (multi-modèles). Probabiliste : CRPS / Quantile Loss pour l'incertitude." },
      { t: "Normalisation", c: "RevIN (Reversible Instance Norm) : Crucial pour gérer le 'Distribution Shift' dans les Transformers." },
      { t: "Architectures SOTA", c: "PatchTST (Patching + Channel Indep), TimesNet (2D Conv), TSMixer (MLP-only). Simple souvent meilleur." },
      { t: "Transformers", c: "Cross-Attention (Variables exogènes), Decoder-only (GPT time-series) vs Encoder-Decoder." }
    ]
  }
};

// --- DATA: 50 QUESTIONS ---
// Types: 'mcq' (QCM), 'open' (Réponse Libre/Auto-éval)
// Difficulty: 'easy', 'medium', 'hard'
// IMPORTANT: La bonne réponse n'est PAS toujours la plus longue !

export const database = [
  // --- SESSION 1 : BASES & AR (10 Q) ---
  {
    id: 1, type: 'mcq', diff: 'medium',
    q: "Quelle est la condition stricte sur le coefficient $\\phi$ pour qu'un processus AR(1) $X_t = \\phi X_{t-1} + \\epsilon_t$ soit stationnaire ?",
    options: ["$\\phi = 1$", "$|\\phi| < 1$", "$\\phi > 0$", "$|\\phi| > 1$", "$\\phi^2 < \\sigma^2$ où $\\sigma^2$ est la variance du bruit"],
    correct: 1,
    expl: "Si $|\\phi| \\ge 1$, la série explose ou devient une marche aléatoire (racine unitaire)."
  },
  {
    id: 2, type: 'open', diff: 'hard',
    q: "Donnez les 3 conditions du 'Triangle de Stationnarité' pour un processus AR(2).",
    answer: "1. $\\phi_1 + \\phi_2 < 1$\n2. $\\phi_2 - \\phi_1 < 1$\n3. $|\\phi_2| < 1$",
    expl: "Ces conditions assurent que les racines du polynôme caractéristique sont en dehors du cercle unité."
  },
  {
    id: 3, type: 'mcq', diff: 'easy',
    q: "Laquelle de ces tâches est supervisée ?",
    options: ["Forecasting", "Clustering de séries", "Réduction de dimension (PCA)", "Anomaly Detection (non supervisée)", "Segmentation temporelle non étiquetée"],
    correct: 0,
    expl: "Le Forecasting utilise les valeurs passées comme 'features' et les valeurs futures comme 'labels' (targets)."
  },
  {
    id: 4, type: 'mcq', diff: 'hard',
    q: "Quelle est l'équation du polynôme caractéristique d'un AR(2) $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\epsilon_t$ ?",
    options: ["$1 - \\phi_1 z - \\phi_2 z^2 = 0$", "$z^2 - \\phi_1 z - \\phi_2 = 0$", "$1 + \\phi_1 z + \\phi_2 z^2 = 0$", "$\\phi_1 z + \\phi_2 z^2 = 1$", "$z^2 + \\phi_1 z + \\phi_2 = 0$ en notation inversée"],
    correct: 0,
    expl: "On écrit $(1 - \\phi_1 L - \\phi_2 L^2)X_t = \\epsilon_t$. Le polynôme est donc $1 - \\phi_1 z - \\phi_2 z^2$."
  },
  {
    id: 5, type: 'mcq', diff: 'medium',
    q: "Que signifie 'i.i.d' pour le bruit $\\epsilon_t$ ?",
    options: ["Indépendant et Identiquement Distribué", "Intégré et Identifié par Différenciation", "Incrémental et Itérativement Dérivé", "Inversement Distribué", "Indéterminé et Incomplet dans la Distribution"],
    correct: 0,
    expl: "C'est l'hypothèse standard pour le terme d'erreur (souvent un bruit blanc gaussien)."
  },
  {
    id: 6, type: 'open', diff: 'medium',
    q: "Expliquez la différence entre Stationnarité 'Stricte' et 'Faible'.",
    answer: "Stricte : Toute la distribution conjointe est invariante par décalage temporel.\nFaible : Seulement les moments d'ordre 1 (moyenne) et 2 (covariance) sont invariants.",
    expl: "En pratique, on vérifie souvent la stationnarité faible car la stricte est trop dure à prouver."
  },
  {
    id: 7, type: 'mcq', diff: 'easy',
    q: "Un Bruit Blanc est-il stationnaire ?",
    options: ["Oui", "Non", "Seulement s'il est Gaussien", "Seulement si sa variance est nulle", "Seulement si sa moyenne est non nulle"],
    correct: 0,
    expl: "Oui, moyenne 0, variance constante $\\sigma^2$, autocorrélations nulles."
  },
  {
    id: 8, type: 'mcq', diff: 'hard',
    q: "Si la fonction d'autocorrélation (ACF) décroît très lentement (linéairement), cela suggère :",
    options: ["Un processus stationnaire", "Mémoire longue ou racine unitaire", "Un bruit blanc", "Une saisonnalité", "Un processus MA(1) avec coefficient proche de 1"],
    correct: 1,
    expl: "C'est le signe d'une 'mémoire longue' ou d'une racine unitaire. Une série stationnaire a une ACF qui décroît vite (exponentiellement)."
  },
  {
    id: 9, type: 'mcq', diff: 'medium',
    q: "Pour évaluer une prévision avec des outliers importants, quelle métrique éviter ?",
    options: ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "Huber Loss", "Quantile Loss", "Median Absolute Deviation"],
    correct: 0,
    expl: "Le carré dans le MSE donne un poids énorme aux outliers, ce qui peut fausser l'entraînement."
  },
  {
    id: 10, type: 'open', diff: 'easy',
    q: "Citez 3 exemples de 'Time Series' dans la vraie vie.",
    answer: "Cours de bourse, ECG (coeur), Météo (température), Trafic web, Consommation électrique...",
    expl: "Toute donnée mesurée séquentiellement dans le temps."
  },

  // --- SESSION 2 : ARCHITECTURES (15 Q) ---
  {
    id: 11, type: 'mcq', diff: 'medium',
    q: "Quel est l'avantage principal des TCN par rapport aux RNN ?",
    options: ["Ils ont une mémoire infinie", "Calcul parallélisable", "Ils n'ont pas de poids à entraîner", "Meilleure gestion du gradient explosif", "Ils capturent mieux les dépendances courtes"],
    correct: 1,
    expl: "Les convolutions peuvent être calculées sur toute la séquence d'un coup, contrairement au RNN qui est séquentiel."
  },
  {
    id: 12, type: 'mcq', diff: 'hard',
    q: "Calcul du champ réceptif (Receptive Field) d'un TCN : Kernel $k=3$, Dilatations $d=[1, 2, 4]$.",
    options: ["8", "15", "7", "12", "21"],
    correct: 1,
    expl: "$RF = 1 + \\sum (k-1)d_i = 1 + 2(1+2+4) = 1 + 2(7) = 15$. Il voit 15 pas de temps."
  },
  {
    id: 13, type: 'open', diff: 'medium',
    q: "Pourquoi utilise-t-on le 'Causal Padding' dans les CNN pour séries temporelles ?",
    answer: "Pour empêcher le modèle de voir le futur (Data Leakage).",
    expl: "Si on veut prédire $t+1$, la convolution en $t$ ne doit utiliser que $t, t-1, t-2...$ d'où le padding uniquement à gauche."
  },
  {
    id: 14, type: 'mcq', diff: 'hard',
    q: "Dans un LSTM, quelle est la fonction d'activation de la 'Cell State Update' (le candidat $\\tilde{C}_t$) ?",
    options: ["Sigmoid $\\sigma$", "Tanh", "ReLU", "Softmax", "LeakyReLU"],
    correct: 1,
    expl: "Tanh est utilisé pour réguler les valeurs entre -1 et 1 avant de les ajouter à la mémoire."
  },
  {
    id: 15, type: 'mcq', diff: 'medium',
    q: "Le phénomène de 'Vanishing Gradient' dans les RNN est dû à :",
    options: ["Multiplications avec valeurs propres $< 1$", "L'utilisation de ReLU", "Un learning rate trop grand", "La taille du batch", "La profondeur du réseau feedforward"],
    correct: 0,
    expl: "Lors de la Backprop Through Time, le gradient est multiplié $T$ fois par $W$. Si $|W| < 1$, ça tend vers 0."
  },
  {
    id: 16, type: 'open', diff: 'hard',
    q: "Quelle est la complexité algorithmique du mécanisme de Self-Attention pour une séquence de longueur $L$ ?",
    answer: "$O(L^2)$ (Quadratique)",
    expl: "Chaque token doit calculer son attention avec tous les autres tokens, créant une matrice $L \\times L$."
  },
  {
    id: 17, type: 'mcq', diff: 'medium',
    q: "À quoi sert le 'Positional Encoding' dans un Transformer ?",
    options: ["Injecter la notion d'ordre", "Compresser l'input", "Normaliser la variance", "Réduire la dimension", "Encoder les relations entre tokens"],
    correct: 0,
    expl: "Le Transformer est invariant par permutation. Sans PE, 'Manger pour Vivre' et 'Vivre pour Manger' seraient vus pareil."
  },
  {
    id: 18, type: 'mcq', diff: 'easy',
    q: "Quel composant d'un ResNet permet d'entraîner des réseaux très profonds ?",
    options: ["Skip Connection", "MaxPooling", "Dropout", "Flatten", "Batch Normalization"],
    correct: 0,
    expl: "Le lien résiduel $x + f(x)$ permet au gradient de 'couler' directement vers les premières couches."
  },
  {
    id: 19, type: 'mcq', diff: 'hard',
    q: "Dans l'attention $A(Q,K,V)$, pourquoi divise-t-on par $\\sqrt{d_k}$ ?",
    options: ["Éviter saturation du Softmax", "Faire une moyenne pondérée", "Normaliser le vecteur de sortie", "Réduire le bruit", "Stabiliser le learning rate"],
    correct: 0,
    expl: "Si le dot product est grand, le Softmax sature et les gradients deviennent minuscules (vanishing)."
  },
  {
    id: 20, type: 'open', diff: 'medium',
    q: "Quelles sont les 3 portes (Gates) d'un LSTM ?",
    answer: "Forget Gate, Input Gate, Output Gate",
    expl: "Elles contrôlent respectivement ce qu'on oublie, ce qu'on écrit, et ce qu'on lit de la mémoire."
  },
  {
    id: 21, type: 'mcq', diff: 'medium',
    q: "Un Autoencodeur (AE) utilisé pour la détection d'anomalies apprend à :",
    options: ["Reconstruire le normal", "Reconstruire les anomalies", "Classifier les anomalies", "Prédire le futur", "Compresser les anomalies"],
    correct: 0,
    expl: "S'il apprend le 'normal', il aura une grande erreur de reconstruction sur les anomalies (qu'il n'a jamais vues)."
  },
  {
    id: 22, type: 'mcq', diff: 'hard',
    q: "Quelle technique permet d'entraîner un RNN de manière plus stable en utilisant les vrais labels précédents au lieu des prédictions ?",
    options: ["Teacher Forcing", "Student Learning", "Gradient Clipping", "Batch Norm", "Curriculum Learning"],
    correct: 0,
    expl: "On nourrit $y_{t-1}$ (vérité) au lieu de $\\hat{y}_{t-1}$ (prédiction) à l'instant $t$."
  },
  {
    id: 23, type: 'open', diff: 'hard',
    q: "Qu'est-ce qu'une convolution 'Dilatée' (Dilated Convolution) ?",
    answer: "Une convolution qui 'saute' des points de l'entrée (ex: trous de taille d) pour agrandir le champ réceptif sans ajouter de paramètres.",
    expl: "Essentiel pour les TCN afin de capturer des dépendances long terme."
  },
  {
    id: 24, type: 'mcq', diff: 'medium',
    q: "Quelle architecture est connue pour être 'Permutation Invariant' sans modifications ?",
    options: ["Transformer (sans PE)", "RNN", "CNN", "LSTM", "GRU bidirectionnel"],
    correct: 0,
    expl: "Le RNN et CNN dépendent de l'ordre ou du voisinage. Le Transformer traite tout l'ensemble globalement (Set processing) sans PE."
  },
  {
    id: 25, type: 'mcq', diff: 'easy',
    q: "Le Dropout sert principalement à :",
    options: ["Réduire l'overfitting", "Accélérer le calcul", "Augmenter les poids", "Visualiser les données", "Améliorer la précision du test"],
    correct: 0,
    expl: "En désactivant des neurones aléatoirement, on empêche la co-adaptation complexe."
  },

  // --- SESSION 3 : SSL & REGULARIZATION (15 Q) ---
  {
    id: 26, type: 'mcq', diff: 'hard',
    q: "La Loss 'InfoNCE' utilisée en Contrastive Learning cherche à maximiser :",
    options: ["Similarité positive vs négatifs", "L'erreur de reconstruction", "La variance du batch", "L'entropie des embeddings", "La distance entre tous les exemples"],
    correct: 0,
    expl: "$L = -\\log \\frac{\\exp(sim^+)}{\\sum \\exp(sim)}$. C'est une cross-entropy catégorielle où la classe correcte est la paire positive."
  },
  {
    id: 27, type: 'open', diff: 'medium',
    q: "Donnez 2 exemples d'Augmentation de données pour les séries temporelles.",
    answer: "Jittering (bruit), Scaling, Time-warping, Cropping, Permutation...",
    expl: "Crucial pour le SSL invariant."
  },
  {
    id: 28, type: 'mcq', diff: 'medium',
    q: "Dans l'analogie du 'Gâteau' de Yann LeCun (Session 3), le 'Reinforcement Learning' est :",
    options: ["La cerise (peu de signal)", "Le gâteau lui-même", "Le glaçage", "L'assiette", "La crème pâtissière"],
    correct: 0,
    expl: "La cerise = Reinforcement (peu de signal). Glaçage = Supervised. Gâteau = Unsupervised/Self-supervised (énorme masse d'information)."
  },
  {
    id: 29, type: 'mcq', diff: 'hard',
    q: "Le 'Dimensional Collapse' en SSL se produit quand :",
    options: ["Représentations identiques", "Le modèle devient trop grand", "La dimension temporelle disparaît", "Le loss devient négatif", "Les embeddings sont trop petits"],
    correct: 0,
    expl: "Le modèle triche en sortant toujours le même vecteur constant, ce qui minimise la distance mais n'apprend rien."
  },
  {
    id: 30, type: 'mcq', diff: 'medium',
    q: "Quelle méthode SSL n'utilise PAS de paires négatives ?",
    options: ["BYOL / SimSiam", "SimCLR", "MoCo", "InfoNCE standard", "Contrastive Predictive Coding"],
    correct: 0,
    expl: "Bootstrap Your Own Latent (BYOL) utilise seulement des paires positives et un réseau 'Teacher' momentum pour éviter le collapse."
  },
  {
    id: 31, type: 'open', diff: 'hard',
    q: "Qu'est-ce que le 'Time-warping' ?",
    answer: "Une augmentation qui déforme l'axe temporel (accélère ou ralentit des segments) pour rendre le modèle robuste aux variations de vitesse.",
    expl: "Similaire au 'Elastic deformation' en vision."
  },
  {
    id: 32, type: 'mcq', diff: 'medium',
    q: "Pour régulariser des séries temporelles, 'DropBlock' est meilleur que 'Dropout' car :",
    options: ["Supprime des plages contiguës", "Il est plus rapide", "Il supprime moins de données", "Il ne modifie pas les poids", "Il s'applique uniquement au test"],
    correct: 0,
    expl: "Les séries temporelles sont très corrélées localement. Supprimer 1 point (Dropout) est facile à interpoler. Supprimer un bloc force à utiliser le contexte lointain."
  },
  {
    id: 33, type: 'mcq', diff: 'easy',
    q: "Le but d'une tâche 'Prétexte' est :",
    options: ["Apprendre de bonnes représentations", "Gagner du temps de calcul", "Réduire le dataset", "Éviter l'overfitting", "Augmenter la vitesse d'inférence"],
    correct: 0,
    expl: "On ne se soucie pas de la performance à la tâche prétexte elle-même, mais de ce que le réseau apprend en la résolvant."
  },
  {
    id: 34, type: 'mcq', diff: 'hard',
    q: "Dans la Triplet Loss $L = max(d(A,P) - d(A,N) + \\alpha, 0)$, que représente $\\alpha$ ?",
    options: ["La marge", "Le learning rate", "L'ancre", "Le nombre de triplets", "La température"],
    correct: 0,
    expl: "On veut que le Négatif soit plus loin du Positif d'au moins cette marge $\\alpha$."
  },
  {
    id: 35, type: 'open', diff: 'medium',
    q: "Pourquoi l'augmentation par 'Permutation' peut-elle être dangereuse pour certaines séries temporelles ?",
    answer: "Elle casse la dépendance temporelle et l'ordre causal.",
    expl: "Si l'ordre exact compte (ex: cause -> effet), mélanger les segments détruit l'information cruciale."
  },

  // --- DIVERS & MIX (10 Q) ---
  {
    id: 36, type: 'mcq', diff: 'hard',
    q: "Un processus 'Random Walk' (Marche Aléatoire) $X_t = X_{t-1} + \\epsilon_t$ est :",
    options: ["Non-stationnaire (Var $\\sim t$)", "Stationnaire", "Convergent vers zéro", "Borné", "Périodique"],
    correct: 0,
    expl: "C'est un processus à racine unitaire (somme cumulée de bruits). Sa variance explose avec le temps."
  },
  {
    id: 37, type: 'mcq', diff: 'medium',
    q: "Le 'Linear Probing' consiste à :",
    options: ["Backbone gelé + couche linéaire", "Entraîner tout le réseau", "Tester sans entraînement", "Fine-tuner toutes les couches", "Geler la dernière couche seulement"],
    correct: 0,
    expl: "C'est la méthode standard pour évaluer la qualité des représentations SSL."
  },
  {
    id: 38, type: 'mcq', diff: 'medium',
    q: "Quelle transformation rend souvent une série financière (prix) stationnaire ?",
    options: ["Différenciation (Returns)", "Le carré", "L'exponentielle", "La somme cumulée", "Le logarithme seul"],
    correct: 0,
    expl: "Les prix suivent souvent une marche aléatoire, mais les rendements (returns) sont souvent stationnaires."
  },
  {
    id: 39, type: 'open', diff: 'hard',
    q: "Quelle est la différence entre 'Model-Agnostic' et 'Model-Specific' interpretability ?",
    answer: "Agnostic : Applicable à tout modèle (ex: SHAP, LIME). Specific : Utilise l'architecture interne (ex: Attention weights).",
    expl: "Pour les Transformers, on visualise souvent les Attention Maps (Specific)."
  },
  {
    id: 40, type: 'mcq', diff: 'hard',
    q: "Si j'ai une série de longueur 1000 et que j'utilise un Transformer standard, la matrice d'attention a une taille :",
    options: ["$1000 \\times 1000$", "$1000 \\times 64$", "$1000 \\times 1$", "$64 \\times 64$", "$512 \\times 512$"],
    correct: 0,
    expl: "$L \\times L$. C'est pourquoi c'est très lourd en mémoire pour les longues séries."
  },
  {
    id: 41, type: 'mcq', diff: 'easy',
    q: "En Data Augmentation, 'Jittering' signifie :",
    options: ["Ajouter du bruit", "Supprimer des points", "Inverser le temps", "Échantillonner aléatoirement", "Normaliser les valeurs"],
    correct: 0,
    expl: "Ajout de bruit aléatoire (souvent gaussien) pour rendre le modèle robuste aux petites variations."
  },
  {
    id: 42, type: 'mcq', diff: 'medium',
    q: "Le 'Early Stopping' se base sur la courbe de :",
    options: ["Validation Loss", "Training Loss", "Test Accuracy", "Training Accuracy", "Gradient Norm"],
    correct: 0,
    expl: "On arrête quand la loss de validation commence à remonter (signe de début d'overfitting), même si la training loss continue de descendre."
  },
  {
    id: 43, type: 'mcq', diff: 'hard',
    q: "Qu'est-ce que le 'Exposure Bias' dans les RNN seq2seq ?",
    options: ["Décalage train/test (auto-régressif)", "La surexposition aux données", "Un biais dans les poids", "Le learning rate inadapté", "La saturation des activations"],
    correct: 0,
    expl: "En train, on donne la vérité. En test, le modèle se nourrit de ses propres erreurs, qui s'accumulent."
  },
  {
    id: 44, type: 'open', diff: 'medium',
    q: "Définissez 'Covariate Shift'.",
    answer: "Changement de la distribution des entrées $P(X)$ entre le train et le test (tandis que $P(Y|X)$ reste stable).",
    expl: "Très fréquent en Time Series (ex: comportement consommateur avant/après Covid)."
  },
  {
    id: 45, type: 'mcq', diff: 'medium',
    q: "Un 'Trend' (Tendance) dans une série temporelle est :",
    options: ["Variation long-terme de la moyenne", "Une variation cyclique courte", "Un bruit aléatoire", "Une erreur de mesure", "Un artefact de l'échantillonnage"],
    correct: 0,
    expl: "Composante déterministe ou stochastique qui indique la direction générale."
  },
  {
    id: 46, type: 'mcq', diff: 'hard',
    q: "Le théorème de décomposition de Wold stipule que tout processus stationnaire peut s'écrire comme :",
    options: ["Déterministe + MA($\\infty$)", "Un AR(1)", "Une somme de sinus", "Une constante", "Un processus ARIMA(1,1,1)"],
    correct: 0,
    expl: "Fondamental en théorie des séries temporelles."
  },
  {
    id: 47, type: 'open', diff: 'hard',
    q: "Pourquoi la Batch Normalization est-elle délicate avec les RNN ?",
    answer: "Car les statistiques de batch changent à chaque pas de temps et la longueur des séquences varie.",
    expl: "On préfère souvent la 'Layer Normalization' pour les RNN/Transformers."
  },
  {
    id: 48, type: 'mcq', diff: 'medium',
    q: "Si mon modèle prédit toujours la moyenne de la série, mon $R^2$ sera proche de :",
    options: ["0", "1", "-1", "0.5", "Indéfini"],
    correct: 0,
    expl: "Le $R^2$ compare la performance par rapport à la prédiction naïve de la moyenne."
  },
  {
    id: 49, type: 'mcq', diff: 'easy',
    q: "Lequel est un framework Python populaire pour les séries temporelles (Deep Learning) ?",
    options: ["PyTorch / TensorFlow", "Excel", "Word", "MATLAB uniquement", "Tableau"],
    correct: 0,
    expl: "Il existe aussi des libs spécifiques comme Darts, PyTorch Forecasting, GluonTS."
  },
  {
    id: 50, type: 'mcq', diff: 'medium',
    q: "L'autocorrélation partielle (PACF) est utile pour identifier l'ordre $p$ d'un processus :",
    options: ["AR(p)", "MA(q)", "ARMA(p,q)", "SARIMA", "GARCH"],
    correct: 0,
    expl: "Pour un AR(p), la PACF se coupe (devient nulle) après le lag $p$."
  },

  // --- NEW ADDITIONS : SESSION 1 EXTENDED ---
  {
    id: 51, type: 'mcq', diff: 'medium',
    q: "Une série temporelle est dite 'faiblement stationnaire' si :",
    options: ["Moyenne variable, Variance cste", "Moyenne linéaire, Variance cste", "Moyenne cste, Cov dépend du lag h", "Distribution Gaussienne parfaite", "Autocorrélation nulle à tous les lags"],
    correct: 2,
    expl: "La covariance $\\gamma(t, t+h)$ ne doit dépendre que de $h$ (la distance temporelle), pas de la position absolue $t$."
  },
  {
    id: 52, type: 'mcq', diff: 'hard',
    q: "Dans une marche aléatoire (Random Walk) $X_{t+1} = X_t + w_t$ :",
    options: ["La variance est constante", "La série est stationnaire", "Variance linéaire avec t", "Elle converge vers 0", "Elle est bornée par $\\sigma$"],
    correct: 2,
    expl: "$Var(X_t) = t \\cdot \\sigma^2$. La variance explose avec le temps, donc non stationnaire."
  },
  {
    id: 53, type: 'mcq', diff: 'hard',
    q: "Différence principale entre Autocorrélation et Autocovariance ?",
    options: ["Autocorr normalisée (-1 à 1)", "L'autocovariance mesure la causalité", "L'autocorrélation est pour les bruits blancs", "Aucune différence", "L'autocovariance est toujours positive"],
    correct: 0,
    expl: "L'autocorrélation est l'autocovariance divisée par la variance (normalisation)."
  },
  {
    id: 54, type: 'mcq', diff: 'medium',
    q: "Le modèle Prophet de Facebook est basé sur une décomposition :",
    options: ["Multiplicative récurrente", "Additive : Tendance + Saison + Fêtes", "Purement AR", "Basée sur les ondelettes", "LSTM avec attention"],
    correct: 1,
    expl: "Prophet modélise : $\\hat{y}(t) = g(t) + s(t) + h(t)$ (Tendance + Saisonnalité + Jours fériés)."
  },
  {
    id: 55, type: 'mcq', diff: 'medium',
    q: "Si l'autocorrélation chute brusquement à 0 après le lag $q$, cela suggère un processus :",
    options: ["AR(p)", "MA(q)", "Random Walk", "Non stationnaire", "AR intégré"],
    correct: 1,
    expl: "Signature théorique d'un processus MA(q). Pour un AR(p), ça décroît exponentiellement."
  },
  {
    id: 56, type: 'mcq', diff: 'easy',
    q: "À quoi sert la différenciation (differencing) $X_t - X_{t-1}$ ?",
    options: ["Lisser la courbe", "Stationnariser la série", "Augmenter le dataset", "Réduire le bruit", "Calculer la tendance"],
    correct: 1,
    expl: "Supprime la tendance linéaire et stabilise la moyenne."
  },
  {
    id: 57, type: 'mcq', diff: 'hard',
    q: "Quel modèle incorpore explicitement des lois physiques (état caché) ?",
    options: ["ARIMA", "LSTM", "Filtre de Kalman", "SVM", "Random Forest"],
    correct: 2,
    expl: "Estime l'état caché d'un système à partir de mesures bruitées (très utilisé GPS)."
  },
  {
    id: 58, type: 'mcq', diff: 'hard',
    q: "Quelle métrique permet de comparer des erreurs sur des séries d'échelles très différentes ?",
    options: ["MSE", "MAE", "MAPE", "RMSE", "R²"],
    correct: 2,
    expl: "MAPE est un pourcentage d'erreur relative, indépendant de l'unité (ex: € vs M€)."
  },

  // --- NEW ADDITIONS : SESSION 2 EXTENDED ---
  {
    id: 59, type: 'mcq', diff: 'medium',
    q: "Différence structurelle majeure entre LSTM et GRU ?",
    options: ["GRU a plus de paramètres", "GRU n'a pas de Cell State", "LSTM a 2 portes, GRU en a 3", "GRU utilise ReLU", "LSTM n'a pas de hidden state"],
    correct: 1,
    expl: "GRU fusionne Cell State et Hidden State, et a 2 portes au lieu de 3."
  },
  {
    id: 60, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Qu'est-ce que 'Channel Independence' dans PatchTST ?",
    options: ["Chaque canal a son modèle", "Variable → même Backbone, poids partagés", "Les canaux sont moyennés", "Suppression des corrélations", "Attention croisée entre canaux"],
    correct: 1,
    expl: "Apprend une structure temporelle globale robuste et réduit le nombre de paramètres."
  },
  {
    id: 61, type: 'mcq', diff: 'hard',
    q: "Dans l'Attention, que sont Q, K, V ?",
    options: ["Quantités, Kernels, Vecteurs", "Questions, Keys, Values", "Queries, Keys, Values", "Quality, Known, Variance", "Quadrants, Keys, Vectors"],
    correct: 2,
    expl: "Analogie base de données : Requête (Query) comparée aux Clés (Keys) pour pondérer les Valeurs (Values)."
  },

  // --- NEW ADDITIONS : SESSION 3 EXTENDED ---
  {
    id: 62, type: 'mcq', diff: 'medium',
    q: "Idée centrale du Self-Supervised Learning (SSL) ?",
    options: ["Apprendre sans aucune donnée", "Données → pseudo-labels", "Superviseur humain", "Synonyme de RL", "Labels aléatoires"],
    correct: 1,
    expl: "On masque une partie de l'input et le modèle doit la deviner (Pretext Task)."
  },
  {
    id: 63, type: 'mcq', diff: 'hard',
    q: "MoCo (Momentum Contrast) utilise un 'Momentum Encoder' pour :",
    options: ["Accélérer le gradient", "Cohérence des clés négatives", "Réduire la mémoire", "Encoder le futur", "Augmenter le batch"],
    correct: 1,
    expl: "Si l'encodeur de clés change trop vite, la loss est instable. Une moyenne glissante stabilise les cibles."
  },
  {
    id: 64, type: 'mcq', diff: 'medium',
    q: "Pourquoi le 'Reverse Time' augmentation est risqué pour un ECG ?",
    options: ["C'est impossible à coder", "ECG a causalité P-QRS-T", "Ça ne change rien", "Le signal devient bruité", "La fréquence change"],
    correct: 1,
    expl: "Inverser le temps pourrait créer une pathologie impossible, faussant l'apprentissage (contrairement à une image)."
  },

  // --- NEW ADDITIONS : SESSION 4 (AVANCÉ / SOTA) ---
  {
    id: 65, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Dans une stratégie de prévision 'Recursive' (Iterative), le risque principal est :",
    options: ["Trop de modèles à gérer", "Accumulation d'erreurs", "Sur-apprentissage", "Temps de calcul excessif", "Sous-apprentissage"],
    correct: 1,
    expl: "Si la première prédiction est fausse, on la réinjecte pour prédire la suite, amplifiant l'erreur."
  },
  {
    id: 66, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Quel est l'avantage de la stratégie 'Direct' pour l'horizon $H$ ?",
    options: ["Un seul modèle pour tout", "Un modèle par pas → pas d'accumulation", "Moins coûteux en calcul", "Plus simple à implémenter", "Meilleure interprétabilité"],
    correct: 1,
    expl: "On entraîne $H$ modèles distincts : $f_1$ pour $t+1$, $f_2$ pour $t+2$... Indépendance des erreurs."
  },
  {
    id: 67, type: 'mcq', diff: 'expert', isBonus: true,
    q: "Le score CRPS (Continuous Ranked Probability Score) sert à évaluer :",
    options: ["Une prédiction ponctuelle", "Une distribution complète", "La classification", "La vitesse d'inférence", "L'erreur de calibration"],
    correct: 1,
    expl: "Il généralise le MAE pour des prédictions probabilistes. Il mesure la distance entre la CDF prédite et la CDF réelle (Step function)."
  },
  {
    id: 68, type: 'mcq', diff: 'expert', isBonus: true,
    q: "Pourquoi 'RevIN' (Reversible Instance Normalization) est-il crucial pour les Transformers ?",
    options: ["Il accélère le GPU", "Gère le Distribution Shift", "Il remplace l'Attention", "Il supprime le bruit", "Il réduit le nombre de paramètres"],
    correct: 1,
    expl: "Les séries temporelles changent de statistiques (moyenne/variance) avec le temps. RevIN supprime ce shift temporairement pour que le modèle se concentre sur la forme."
  },
  {
    id: 69, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Dans PatchTST, quel est l'intérêt principal du 'Patching' ?",
    options: ["Augmenter la résolution", "Réduit L, capture contexte local", "Supprimer les NaN", "Compresser les données", "Ajouter du padding"],
    correct: 1,
    expl: "Regrouper les points (ex: 16 points = 1 token) réduit la complexité quadratique de l'Attention et donne du contexte local."
  },
  {
    id: 70, type: 'mcq', diff: 'expert', isBonus: true,
    q: "L'architecture 'TimesNet' transforme la série 1D en 2D pour :",
    options: ["L'afficher graphiquement", "Conv 2D sur périodes dominantes", "Réduire la mémoire", "Faire de l'augmentation", "Appliquer du pooling spatial"],
    correct: 1,
    expl: "Elle analyse les variabilités intra-période et inter-période simultanément."
  },
  {
    id: 71, type: 'mcq', diff: 'hard', isBonus: true,
    q: "TSMixer a prouvé que :",
    options: ["Les Transformers sont indispensables", "MLPs simples battent Transformers", "Les RNN sont le futur", "Le Deep Learning est inutile", "L'attention est obligatoire"],
    correct: 1,
    expl: "Architecture 100% MLP (Multi-Layer Perceptron) qui atteint des performances SOTA avec moins de calcul."
  },
  {
    id: 72, type: 'mcq', diff: 'medium', isBonus: true,
    q: "Quelle Loss Function utiliser pour prédire un intervalle de confiance (ex: 90%) ?",
    options: ["MSE", "Cross-Entropy", "Quantile Loss (Pinball)", "L1 Loss", "Hinge Loss"],
    correct: 2,
    expl: "La Quantile Loss pénalise asymétriquement les erreurs pour forcer le modèle à prédire un percentile donné (ex: q=0.9)."
  },
  {
    id: 73, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Quand utiliser la Cross-Attention dans un Transformer temporel ?",
    options: ["Toujours", "Variables exogènes dans décodeur", "Pour la classification", "Jamais", "Seulement pour les images"],
    correct: 1,
    expl: "Le décodeur utilise la sortie de l'encodeur (ou des variables externes) comme 'Keys' et 'Values' via la Cross-Attention."
  },
  {
    id: 74, type: 'mcq', diff: 'medium', isBonus: true,
    q: "Différence entre 'Point Forecast' et 'Probabilistic Forecast' ?",
    options: ["Aucune", "Point = valeur; Prob = distribution", "Point = futur proche", "Point = classification", "Prob = plus rapide"],
    correct: 1,
    expl: "La prévision probabiliste est essentielle pour la gestion des risques (stocks, finance)."
  },
  {
    id: 75, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Le concept de 'Lookback Window' désigne :",
    options: ["Une technique de debug", "Fenêtre d'historique en entrée", "La fenêtre de prédiction", "Le temps d'entraînement", "La validation croisée"],
    correct: 1,
    expl: "C'est la taille de la séquence d'entrée $L$. Un ratio typique par rapport à l'horizon $H$ est souvent $L \\approx 1.5H$ à $4H$."
  },
  {
    id: 76, type: 'mcq', diff: 'expert', isBonus: true,
    q: "Dans un modèle 'Decoder-only' (type GPT) pour séries temporelles :",
    options: ["Pas de prévision possible", "Masque d'attention triangulaire", "On voit le futur", "Encodeur obligatoire", "Cross-attention requise"],
    correct: 1,
    expl: "Le masque causal empêche le modèle de voir les tokens futurs. Idéal pour la génération (Forecasting génératif)."
  },
  {
    id: 77, type: 'mcq', diff: 'medium', isBonus: true,
    q: "Pourquoi normaliser chaque fenêtre (Instance Norm) indépendamment est risqué ?",
    options: ["Efface l'amplitude (Scale)", "C'est trop lent", "Instabilité numérique", "Perte de gradients", "Overfitting"],
    correct: 0,
    expl: "Si on supprime la moyenne/variance de la fenêtre, on perd l'info de niveau global. RevIN réinjecte cette info à la fin."
  },
  {
    id: 78, type: 'mcq', diff: 'hard', isBonus: true,
    q: "Qu'est-ce que le 'Teacher Forcing' ?",
    options: ["Apprentissage supervisé renforcé", "Utilise $y_{t-1}$ vrai au lieu de $\\hat{y}_{t-1}$", "Forcer le modèle à converger", "Une régularisation L2", "Technique de data augmentation"],
    correct: 1,
    expl: "Stabilisation de l'entraînement des RNN/Seq2Seq."
  },
  {
    id: 79, type: 'mcq', diff: 'expert', isBonus: true,
    q: "Quelle métrique favorise la parcimonie (Sparsity) ?",
    options: ["L2 (Ridge)", "L1 (Lasso)", "Dropout", "Adam", "Elastic Net avec $\\alpha = 0$"],
    correct: 1,
    expl: "La régularisation L1 tend à annuler les coefficients des features inutiles."
  }
];
