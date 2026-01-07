import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  BookOpen, CheckCircle, XCircle, Brain, ArrowRight, RotateCcw, 
  Award, Zap, Calculator, Activity, Shuffle, List, AlertTriangle, 
  Layers, Eye, MessageSquare, ChevronDown, ChevronUp, FileText, 
  Thermometer, Anchor, Network
} from 'lucide-react';

// --- GESTION DU LATEX (KATEX) ---
const useKatex = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    if (window.katex) { setIsLoaded(true); return; }
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js";
    script.async = true;
    script.onload = () => setIsLoaded(true);
    document.body.appendChild(script);
    const link = document.createElement('link');
    link.href = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css";
    link.rel = "stylesheet";
    document.head.appendChild(link);
    return () => { document.body.removeChild(script); document.head.removeChild(link); };
  }, []);
  return isLoaded;
};

const LatexText = ({ text, className = "" }) => {
  const isKatexLoaded = useKatex();
  const containerRef = useRef(null);
  useEffect(() => {
    if (!isKatexLoaded || !containerRef.current) return;
    const parts = text.split('$');
    const container = containerRef.current;
    container.innerHTML = '';
    parts.forEach((part, index) => {
      const span = document.createElement('span');
      if (index % 2 === 1) {
        try { window.katex.render(part, span, { throwOnError: false, displayMode: false }); } 
        catch (e) { span.innerText = `$${part}$`; }
      } else { span.innerText = part; }
      container.appendChild(span);
    });
  }, [text, isKatexLoaded]);
  if (!isKatexLoaded) return <span className={className}>{text}</span>;
  return <span ref={containerRef} className={className} />;
};

// --- DATA: FLASHCARDS (CONCEPTS CLÉS) ---
const flashcards = [
  {
    id: 1,
    title: "Le Triangle de Stationnarité AR(2)",
    content: "Pour $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\epsilon_t$. \nLes racines du polynôme $(1 - \\phi_1 z - \\phi_2 z^2)$ doivent être hors du cercle unité.\nConditions strictes :\n1. $\\phi_1 + \\phi_2 < 1$\n2. $\\phi_2 - \\phi_1 < 1$\n3. $|\\phi_2| < 1$",
    icon: "triangle"
  },
  {
    id: 2,
    title: "Receptive Field (TCN)",
    content: "Contrairement aux RNN, la mémoire d'un TCN est finie. Elle dépend de la taille du noyau $k$, du nombre de couches $L$ et des facteurs de dilatation $d_i$.\nFormule du champ réceptif :\n$R = 1 + \\sum_{i=0}^{L-1} (k-1) \\cdot d_i$\nSi dilatation exponentielle ($d=2^i$) : $R \\approx 2^L$.",
    icon: "metric"
  },
  {
    id: 3,
    title: "Attention Mechanism (Maths)",
    content: "Le cœur du Transformer. On calcule une similarité entre Query ($Q$) et Key ($K$).\n$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$\nLe facteur $\\frac{1}{\\sqrt{d_k}}$ sert à éviter que le produit scalaire n'explose (ce qui tuerait les gradients du Softmax).",
    icon: "eye"
  },
  {
    id: 4,
    title: "Vanishing Gradient (RNN)",
    content: "En BPTT (Backprop Through Time), le gradient est multiplié par la matrice de poids récurrente $W_h$ à chaque pas.\nSi la plus grande valeur propre $|\\lambda| < 1$, le gradient tend vers 0 exponentiellement ($0.9^{100} \\approx 0$).\nSolution : LSTM/GRU (additif via gates) ou initialisation orthogonale.",
    icon: "network"
  },
  {
    id: 5,
    title: "SSL : InfoNCE Loss",
    content: "Utilisée en Contrastive Learning.\n$L = -\\log \\frac{\\exp(sim(z_i, z_j)/\\tau)}{\\sum_{k} \\exp(sim(z_i, z_k)/\\tau)}$\nOn veut maximiser la probabilité de la paire positive $(i,j)$ par rapport à toutes les paires négatives $k$ du batch. $\\tau$ est la température.",
    icon: "mask"
  },
  {
    id: 6,
    title: "Dimensional Collapse",
    content: "Problème en SSL où le modèle 'triche' en projetant toutes les données dans un sous-espace très réduit (ou un point constant). \nConséquence : Bonne loss mais représentations inutiles.\nSolutions : Contrastive negative pairs, Asymmetric networks (SimSiam/BYOL), Variance-Invariance-Covariance regularization.",
    icon: "chart"
  }
];

// --- DATA: SLIDE SUMMARIES (CHEAT SHEETS) ---
const slideSummaries = {
  session1: {
    title: "Session 1 : Concepts & Tâches",
    points: [
      { t: "Définition", c: "Série Temporelle : Suite d'observations indexées par le temps $X = \\{X_{t_1}, ..., X_{t_n}\\}$. Processus Stochastique : La loi de probabilité génératrice." },
      { t: "Stationnarité", c: "Faible : Espérance constante, Variance constante, Autocovariance ne dépend que du délai $h$ (pas du temps $t$)." },
      { t: "Tâches", c: "Forecasting (Prédire $X_{t+h}$), Classification (Labeliser la série), Event Detection (Trouver $t$ anormal), Imputation (Remplir les trous)." },
      { t: "Métriques", c: "MSE (Sensible outliers), MAE (Robuste), MAPE (Pourcentage, attention si $y=0$)." }
    ]
  },
  session2: {
    title: "Session 2 : Architectures DNN",
    points: [
      { t: "RNN & Problèmes", c: "Traitement séquentiel. Problème : Vanishing Gradient sur longues séquences (BPTT). Solution : LSTM/GRU (Gating)." },
      { t: "CNN (TCN)", c: "Convolutions 1D. Causales (pas de futur) + Dilatées (grand champ réceptif). Parallélisables $\\to$ Rapides." },
      { t: "Transformers", c: "Mécanisme d'Attention $O(L^2)$. Invariant par permutation $\\to$ Besoin de Positional Encoding. SOTA actuel mais lourd." },
      { t: "ResNets", c: "Skip connections $y = f(x) + x$ pour éviter la dégradation du gradient dans les réseaux profonds." }
    ]
  },
  session3: {
    title: "Session 3 : Self-Supervised Learning",
    points: [
      { t: "Le Gâteau de LeCun", c: "Reinforcement (Cerise), Supervised (Glaçage), Unsupervised (Le Gâteau lui-même). Le SSL est crucial car les labels sont rares." },
      { t: "Contrastive Learning", c: "Rapprocher $(x, x^+)$ (Positifs) et éloigner $(x, x^-)$ (Négatifs). Loss : InfoNCE. Augmentations cruciales (Crop, Noise)." },
      { t: "Evaluation SSL", c: "Linear Probing (Entraîner juste un classifieur linéaire sur les embeddings gelés) ou Zero-Shot k-NN." },
      { t: "Dimensional Collapse", c: "Quand les embeddings sont redondants. Mesuré par le rang de la matrice de covariance." }
    ]
  }
};

// --- DATA: 50 QUESTIONS ---
// Types: 'mcq' (QCM), 'open' (Réponse Libre/Auto-éval)
// Difficulty: 'easy', 'medium', 'hard'

const database = [
  // --- SESSION 1 : BASES & AR (10 Q) ---
  {
    id: 1, type: 'mcq', diff: 'medium',
    q: "Quelle est la condition stricte sur le coefficient $\\phi$ pour qu'un processus AR(1) $X_t = \\phi X_{t-1} + \\epsilon_t$ soit stationnaire ?",
    options: ["$\\phi = 1$", "$|\\phi| < 1$", "$\\phi > 0$", "$|\\phi| > 1$"],
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
    options: ["Forecasting", "Clustering de séries", "Réduction de dimension (PCA)", "Anomaly Detection (non supervisée)"],
    correct: 0,
    expl: "Le Forecasting utilise les valeurs passées comme 'features' et les valeurs futures comme 'labels' (targets)."
  },
  {
    id: 4, type: 'mcq', diff: 'hard',
    q: "Quelle est l'équation du polynôme caractéristique d'un AR(2) $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\epsilon_t$ ?",
    options: ["$1 - \\phi_1 z - \\phi_2 z^2 = 0$", "$z^2 - \\phi_1 z - \\phi_2 = 0$", "$1 + \\phi_1 z + \\phi_2 z^2 = 0$", "$\\phi_1 z + \\phi_2 z^2 = 1$"],
    correct: 0,
    expl: "On écrit $(1 - \\phi_1 L - \\phi_2 L^2)X_t = \\epsilon_t$. Le polynôme est donc $1 - \\phi_1 z - \\phi_2 z^2$."
  },
  {
    id: 5, type: 'mcq', diff: 'medium',
    q: "Que signifie 'i.i.d' pour le bruit $\\epsilon_t$ ?",
    options: ["Indépendant et Identiquement Distribué", "Immédiatement Indépendant de la Distribution", "Intégré et Induit", "Inversement Distribué"],
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
    options: ["Oui", "Non", "Seulement s'il est Gaussien", "Seulement la nuit"],
    correct: 0,
    expl: "Oui, moyenne 0, variance constante $\\sigma^2$, autocorrélations nulles."
  },
  {
    id: 8, type: 'mcq', diff: 'hard',
    q: "Si la fonction d'autocorrélation (ACF) décroît très lentement (linéairement), cela suggère :",
    options: ["Un processus stationnaire", "Un processus non-stationnaire (ex: Marche Aléatoire)", "Un bruit blanc", "Une saisonnalité"],
    correct: 1,
    expl: "C'est le signe d'une 'mémoire longue' ou d'une racine unitaire. Une série stationnaire a une ACF qui décroît vite (exponentiellement)."
  },
  {
    id: 9, type: 'mcq', diff: 'medium',
    q: "Pour évaluer une prévision avec des outliers importants, quelle métrique éviter ?",
    options: ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "Huber Loss", "Quantile Loss"],
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
    options: ["Ils ont une mémoire infinie", "Ils sont parallélisables (calcul rapide)", "Ils n'ont pas de poids", "Ils sont plus vieux"],
    correct: 1,
    expl: "Les convolutions peuvent être calculées sur toute la séquence d'un coup, contrairement au RNN qui est séquentiel."
  },
  {
    id: 12, type: 'mcq', diff: 'hard',
    q: "Calcul du champ réceptif (Receptive Field) d'un TCN : Kernel $k=3$, Dilatations $d=[1, 2, 4]$.",
    options: ["8", "15", "7", "12"],
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
    options: ["Sigmoid $\\sigma$", "Tanh", "ReLU", "Softmax"],
    correct: 1,
    expl: "Tanh est utilisé pour réguler les valeurs entre -1 et 1 avant de les ajouter à la mémoire."
  },
  {
    id: 15, type: 'mcq', diff: 'medium',
    q: "Le phénomène de 'Vanishing Gradient' dans les RNN est dû à :",
    options: ["Des multiplications répétées de matrices avec valeurs propres $< 1$", "L'utilisation de ReLU", "Un learning rate trop grand", "La taille du batch"],
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
    options: ["À injecter la notion d'ordre séquentiel", "À compresser l'input", "À normaliser la variance", "À rien"],
    correct: 0,
    expl: "Le Transformer est invariant par permutation. Sans PE, 'Manger pour Vivre' et 'Vivre pour Manger' seraient vus pareil."
  },
  {
    id: 18, type: 'mcq', diff: 'easy',
    q: "Quel composant d'un ResNet permet d'entraîner des réseaux très profonds ?",
    options: ["Skip Connection (Residual Link)", "MaxPooling", "Dropout", "Flatten"],
    correct: 0,
    expl: "Le lien résiduel $x + f(x)$ permet au gradient de 'couler' directement vers les premières couches."
  },
  {
    id: 19, type: 'mcq', diff: 'hard',
    q: "Dans l'attention $A(Q,K,V)$, pourquoi divise-t-on par $\\sqrt{d_k}$ ?",
    options: ["Pour éviter que le produit scalaire soit trop grand (gradients nuls)", "Pour faire une moyenne", "C'est une constante magique", "Pour réduire le bruit"],
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
    options: ["Reconstruire les données normales", "Reconstruire les anomalies", "Classifier les anomalies", "Prédire le futur"],
    correct: 0,
    expl: "S'il apprend le 'normal', il aura une grande erreur de reconstruction sur les anomalies (qu'il n'a jamais vues)."
  },
  {
    id: 22, type: 'mcq', diff: 'hard',
    q: "Quelle technique permet d'entraîner un RNN de manière plus stable en utilisant les vrais labels précédents au lieu des prédictions ?",
    options: ["Teacher Forcing", "Student Learning", "Gradient Clipping", "Batch Norm"],
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
    options: ["Transformer (sans PE)", "RNN", "CNN", "LSTM"],
    correct: 0,
    expl: "Le RNN et CNN dépendent de l'ordre ou du voisinage. Le Transformer traite tout l'ensemble globalement (Set processing) sans PE."
  },
  {
    id: 25, type: 'mcq', diff: 'easy',
    q: "Le Dropout sert principalement à :",
    options: ["Réduire l'overfitting", "Accélérer le calcul", "Augmenter les poids", "Visualiser les données"],
    correct: 0,
    expl: "En désactivant des neurones aléatoirement, on empêche la co-adaptation complexe."
  },

  // --- SESSION 3 : SSL & REGULARIZATION (15 Q) ---
  {
    id: 26, type: 'mcq', diff: 'hard',
    q: "La Loss 'InfoNCE' utilisée en Contrastive Learning cherche à maximiser :",
    options: ["La similarité avec l'exemple positif par rapport aux négatifs", "L'erreur de reconstruction", "La variance du batch", "L'entropie"],
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
    options: ["La cerise sur le gâteau (peu de bits de feedback)", "Le gâteau lui-même", "Le glaçage", "L'assiette"],
    correct: 0,
    expl: "La cerise = Reinforcement (peu de signal). Glaçage = Supervised. Gâteau = Unsupervised/Self-supervised (énorme masse d'information)."
  },
  {
    id: 29, type: 'mcq', diff: 'hard',
    q: "Le 'Dimensional Collapse' en SSL se produit quand :",
    options: ["Toutes les représentations deviennent identiques ou occupent un sous-espace réduit", "Le modèle devient trop grand", "La dimension temporelle disparait", "Le loss devient négatif"],
    correct: 0,
    expl: "Le modèle triche en sortant toujours le même vecteur constant, ce qui minimise la distance mais n'apprend rien."
  },
  {
    id: 30, type: 'mcq', diff: 'medium',
    q: "Quelle méthode SSL n'utilise PAS de paires négatives ?",
    options: ["BYOL / SimSiam", "SimCLR", "MoCo", "InfoNCE"],
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
    options: ["Il supprime des plages temporelles contiguës", "Il est plus rapide", "Il supprime tout", "Il ne fait rien"],
    correct: 0,
    expl: "Les séries temporelles sont très corrélées localement. Supprimer 1 point (Dropout) est facile à interpoler. Supprimer un bloc force à utiliser le contexte lointain."
  },
  {
    id: 33, type: 'mcq', diff: 'easy',
    q: "Le but d'une tâche 'Prétexte' est :",
    options: ["D'apprendre de bonnes représentations", "De gagner du temps", "De faire de la prédiction boursière", "D'éviter d'utiliser un GPU"],
    correct: 0,
    expl: "On ne se soucie pas de la performance à la tâche prétexte elle-même, mais de ce que le réseau apprend en la résolvant."
  },
  {
    id: 34, type: 'mcq', diff: 'hard',
    q: "Dans la Triplet Loss $L = max(d(A,P) - d(A,N) + \\alpha, 0)$, que représente $\\alpha$ ?",
    options: ["La marge", "Le learning rate", "L'ancre", "Le nombre de triplets"],
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
    options: ["Non-stationnaire (Variance $\\sim t$)", "Stationnaire", "Convergent", "Borné"],
    correct: 0,
    expl: "C'est un processus à racine unitaire (somme cumulée de bruits). Sa variance explose avec le temps."
  },
  {
    id: 37, type: 'mcq', diff: 'medium',
    q: "Le 'Linear Probing' consiste à :",
    options: ["Geler le backbone pré-entraîné et entraîner juste une couche linéaire finale", "Entraîner tout le réseau", "Tester avec une règle linéaire", "Dessiner des lignes"],
    correct: 0,
    expl: "C'est la méthode standard pour évaluer la qualité des représentations SSL."
  },
  {
    id: 38, type: 'mcq', diff: 'medium',
    q: "Quelle transformation rend souvent une série financière (prix) stationnaire ?",
    options: ["La différenciation (Returns $P_t - P_{t-1}$)", "Le carré", "L'exponentielle", "La somme cumulée"],
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
    options: ["$1000 \\times 1000$ (1 Million)", "$1000 \\times 64$", "$1000 \\times 1$", "$64 \\times 64$"],
    correct: 0,
    expl: "$L \\times L$. C'est pourquoi c'est très lourd en mémoire pour les longues séries."
  },
  {
    id: 41, type: 'mcq', diff: 'easy',
    q: "En Data Augmentation, 'Jittering' signifie :",
    options: ["Ajouter du bruit", "Supprimer des points", "Tourner l'image", "Inverser le temps"],
    correct: 0,
    expl: "Ajout de bruit aléatoire (souvent gaussien) pour rendre le modèle robuste aux petites variations."
  },
  {
    id: 42, type: 'mcq', diff: 'medium',
    q: "Le 'Early Stopping' se base sur la courbe de :",
    options: ["Validation Loss", "Training Loss", "Test Accuracy", "Training Accuracy"],
    correct: 0,
    expl: "On arrête quand la loss de validation commence à remonter (signe de début d'overfitting), même si la training loss continue de descendre."
  },
  {
    id: 43, type: 'mcq', diff: 'hard',
    q: "Qu'est-ce que le 'Exposure Bias' dans les RNN seq2seq ?",
    options: ["Le décalage entre l'entraînement (Teacher Forcing) et le test (Auto-régressif)", "La surexposition aux UV", "Un biais de données", "Le fait de voir trop de données"],
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
    options: ["Une variation long-terme de la moyenne", "Une variation cyclique", "Un bruit aléatoire", "Une erreur de mesure"],
    correct: 0,
    expl: "Composante déterministe ou stochastique qui indique la direction générale."
  },
  {
    id: 46, type: 'mcq', diff: 'hard',
    q: "Le théorème de décomposition de Wold stipule que tout processus stationnaire peut s'écrire comme :",
    options: ["Une somme d'une composante déterministe et d'un processus MA($\\infty$)", "Un AR(1)", "Une somme de sinus", "Une constante"],
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
    options: ["0", "1", "-1", "0.5"],
    correct: 0,
    expl: "Le $R^2$ compare la performance par rapport à la prédiction naïve de la moyenne."
  },
  {
    id: 49, type: 'mcq', diff: 'easy',
    q: "Lequel est un framework Python populaire pour les séries temporelles (Deep Learning) ?",
    options: ["PyTorch / TensorFlow", "Excel", "Word", "Paint"],
    correct: 0,
    expl: "Il existe aussi des libs spécifiques comme Darts, PyTorch Forecasting, GluonTS."
  },
  {
    id: 50, type: 'mcq', diff: 'medium',
    q: "L'autocorrélation partielle (PACF) est utile pour identifier l'ordre $p$ d'un processus :",
    options: ["AR(p)", "MA(q)", "ARMA", "RNN"],
    correct: 0,
    expl: "Pour un AR(p), la PACF se coupe (devient nulle) après le lag $p$."
  }
];

// --- LOGIQUE SHUFFLE ---
const shuffleArray = (array) => {
  const newArr = [...array];
  for (let i = newArr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArr[i], newArr[j]] = [newArr[j], newArr[i]];
  }
  return newArr;
};

// --- COMPOSANT PRINCIPAL ---
export default function RevisionUltimate() {
  useKatex();

  // STATES
  const [appState, setAppState] = useState('menu'); // menu, quiz, results, cheatsheet
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [currentQIndex, setCurrentQIndex] = useState(0);
  
  // Quiz Interaction States
  const [selectedOption, setSelectedOption] = useState(null); // Pour QCM
  const [isRevealed, setIsRevealed] = useState(false); // Pour Open Q
  const [isCorrect, setIsCorrect] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showFlashcard, setShowFlashcard] = useState(false); // Pour les interruptions Flashcard
  
  const [score, setScore] = useState(0);
  const [history, setHistory] = useState([]);
  
  // Config
  const [mode, setMode] = useState('sequential'); // sequential, random, hard

  // START QUIZ
  const startQuiz = (selectedMode) => {
    setMode(selectedMode);
    let q = [...database];
    
    if (selectedMode === 'hard') {
      q = q.filter(item => item.diff === 'hard');
    }
    
    if (selectedMode === 'random' || selectedMode === 'hard') {
      q = shuffleArray(q);
    }
    
    // Pour séquentiel, on garde l'ordre des IDs 1->50
    if (selectedMode === 'sequential') {
      q.sort((a,b) => a.id - b.id);
    }

    setQuizQuestions(q);
    setScore(0);
    setCurrentQIndex(0);
    setHistory([]);
    setAppState('quiz');
    resetQuestionState();
  };

  const resetQuestionState = () => {
    setSelectedOption(null);
    setIsRevealed(false);
    setIsCorrect(null);
    setShowExplanation(false);
    setShowFlashcard(false);
  };

  // --- HANDLERS QCM ---
  const handleMCQOptionClick = (index) => {
    if (selectedOption !== null) return;
    const currentQ = quizQuestions[currentQIndex];
    const correct = index === currentQ.correct;
    
    setSelectedOption(index);
    setIsCorrect(correct);
    setShowExplanation(true);
    
    if (correct) setScore(s => s + 1);
    
    setHistory([...history, { 
      id: currentQ.id,
      q: currentQ.q, 
      correct, 
      type: 'mcq'
    }]);
  };

  // --- HANDLERS OPEN QUESTION ---
  const handleOpenReveal = () => {
    setIsRevealed(true);
  };

  const handleOpenSelfEval = (userJudgeCorrect) => {
    if (isCorrect !== null) return; // Déjà voté
    setIsCorrect(userJudgeCorrect);
    setShowExplanation(true);
    
    if (userJudgeCorrect) setScore(s => s + 1);
    
    setHistory([...history, { 
      id: quizQuestions[currentQIndex].id,
      q: quizQuestions[currentQIndex].q, 
      correct: userJudgeCorrect, 
      type: 'open'
    }]);
  };

  // LOGIQUE FLASHCARD
  const handleNext = () => {
    const nextIndex = currentQIndex + 1;

    // Si on est en mode 'sequential' ou 'random', on affiche une flashcard toutes les 5 questions
    // En mode 'hard', on peut désactiver ou non (ici on garde)
    if (nextIndex % 5 === 0 && nextIndex < quizQuestions.length && !showFlashcard) {
      setShowFlashcard(true);
    } else {
      advanceQuestion();
    }
  };

  const advanceQuestion = () => {
    const nextIndex = currentQIndex + 1;
    if (nextIndex < quizQuestions.length) {
      setCurrentQIndex(nextIndex);
      resetQuestionState();
    } else {
      setAppState('results');
    }
  };

  // --- UI PARTS ---

  // 1. MENU
  if (appState === 'menu') {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-6 font-sans">
        <div className="bg-white rounded-3xl shadow-xl p-8 max-w-lg w-full text-center border border-slate-200">
          <div className="flex justify-center mb-6">
             <div className="bg-blue-600 p-4 rounded-2xl shadow-lg rotate-3">
               <Layers className="w-10 h-10 text-white" />
             </div>
          </div>
          <h1 className="text-3xl font-black text-slate-800 mb-2 tracking-tight">DLST <span className="text-blue-600">Ultimate</span></h1>
          <p className="text-slate-500 mb-8 font-medium">Préparation complète à l'examen Deep Learning Time Series.</p>

          <div className="grid gap-3 mb-8">
            <MenuButton 
              icon={<List className="w-5 h-5 text-indigo-500" />} 
              title="Mode Séquentiel" 
              subtitle="50 questions dans l'ordre du cours"
              onClick={() => startQuiz('sequential')} 
            />
            <MenuButton 
              icon={<Shuffle className="w-5 h-5 text-purple-500" />} 
              title="Mode Aléatoire" 
              subtitle="Mélange complet pour tester vos réflexes"
              onClick={() => startQuiz('random')} 
            />
            <MenuButton 
              icon={<AlertTriangle className="w-5 h-5 text-red-500" />} 
              title="Mode Hardcore" 
              subtitle="Seulement les 20 questions difficiles"
              onClick={() => startQuiz('hard')} 
            />
          </div>

          <button 
            onClick={() => setAppState('cheatsheet')}
            className="w-full py-3 rounded-xl border-2 border-slate-200 text-slate-600 font-bold hover:bg-slate-50 flex items-center justify-center gap-2 transition-colors"
          >
            <FileText className="w-5 h-5" /> Accéder aux Fiches de Révision
          </button>
        </div>
      </div>
    );
  }

  // 2. CHEAT SHEET
  if (appState === 'cheatsheet') {
    return (
      <div className="min-h-screen bg-slate-100 p-4 md:p-8 font-sans">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center gap-4 mb-8">
            <button onClick={() => setAppState('menu')} className="p-2 bg-white rounded-full shadow hover:bg-slate-50 transition-all">
              <RotateCcw className="w-6 h-6 text-slate-600" />
            </button>
            <h1 className="text-2xl font-bold text-slate-800">Fiches de Révision Express</h1>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
             {Object.entries(slideSummaries).map(([key, data]) => (
               <div key={key} className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col hover:shadow-md transition-shadow">
                 <div className="bg-slate-800 p-4 border-b border-slate-700">
                   <h2 className="text-white font-bold text-lg">{data.title}</h2>
                 </div>
                 <div className="p-5 flex-1 bg-gradient-to-b from-white to-slate-50">
                   <ul className="space-y-4">
                     {data.points.map((pt, idx) => (
                       <li key={idx} className="text-sm text-slate-700">
                         <div className="flex items-center gap-2 mb-1">
                           <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                           <strong className="block text-blue-700">{pt.t}</strong>
                         </div>
                         <div className="pl-3.5 text-slate-600 leading-relaxed">
                            <LatexText text={pt.c} />
                         </div>
                       </li>
                     ))}
                   </ul>
                 </div>
               </div>
             ))}
          </div>
        </div>
      </div>
    );
  }

  // 3. RESULTS
  if (appState === 'results') {
    const percentage = Math.round((score / quizQuestions.length) * 100);
    return (
      <div className="min-h-screen bg-slate-50 p-6 flex items-center justify-center font-sans">
        <div className="bg-white rounded-3xl shadow-xl p-8 max-w-2xl w-full border border-slate-200">
          <div className="text-center mb-8">
             <div className="inline-block p-4 rounded-full bg-yellow-100 mb-4">
               <Award className="w-12 h-12 text-yellow-600" />
             </div>
             <h2 className="text-3xl font-bold text-slate-800">Terminé !</h2>
             <div className="mt-4 flex flex-col items-center">
               <span className={`text-6xl font-black ${percentage >= 50 ? 'text-green-500' : 'text-red-500'}`}>
                 {percentage}%
               </span>
               <span className="text-slate-400 font-medium mt-2">Score: {score} / {quizQuestions.length}</span>
             </div>
          </div>

          <div className="bg-slate-50 rounded-xl border border-slate-200 max-h-80 overflow-y-auto mb-8 p-2 custom-scrollbar">
            {history.map((h, i) => (
              <div key={i} className="flex items-start gap-3 p-3 border-b border-slate-200 last:border-0 hover:bg-white transition-colors rounded-lg">
                <div className={`mt-1 ${h.correct ? 'text-green-500' : 'text-red-500'}`}>
                  {h.correct ? <CheckCircle size={18} /> : <XCircle size={18} />}
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-1">
                     <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                       Question {i+1} • {h.type === 'mcq' ? 'QCM' : 'Auto-Eval'}
                     </span>
                  </div>
                  <p className="text-sm font-medium text-slate-700 line-clamp-2">
                    <LatexText text={h.q} />
                  </p>
                </div>
              </div>
            ))}
          </div>

          <button onClick={() => setAppState('menu')} className="w-full bg-slate-900 text-white py-4 rounded-xl font-bold hover:bg-slate-800 transition-all flex justify-center items-center gap-2">
            <RotateCcw className="w-5 h-5" /> Retour au Menu
          </button>
        </div>
      </div>
    );
  }

  // 4. FLASHCARD INTERRUPT
  if (showFlashcard) {
     const cardIndex = (Math.floor(currentQIndex / 5)) % flashcards.length;
     const card = flashcards[cardIndex];
     return (
       <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4 font-sans animate-in fade-in duration-300">
         <div className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white rounded-3xl p-8 max-w-lg w-full shadow-2xl relative overflow-hidden border border-slate-700">
            {/* Decorative */}
            <div className="absolute top-[-50px] right-[-50px] w-40 h-40 bg-blue-500 rounded-full blur-3xl opacity-20 pointer-events-none"></div>
            
            <div className="relative z-10 text-center">
              <div className="bg-white/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 backdrop-blur-md border border-white/10">
                 {card.icon === 'triangle' && <Activity className="w-8 h-8 text-yellow-400" />}
                 {card.icon === 'metric' && <Calculator className="w-8 h-8 text-cyan-400" />}
                 {card.icon === 'eye' && <BookOpen className="w-8 h-8 text-pink-400" />}
                 {card.icon === 'network' && <Network className="w-8 h-8 text-green-400" />}
                 {card.icon === 'mask' && <Brain className="w-8 h-8 text-purple-400" />}
                 {card.icon === 'chart' && <Activity className="w-8 h-8 text-orange-400" />}
              </div>
              
              <h3 className="text-xs font-bold tracking-[0.2em] text-blue-300 uppercase mb-2">Flashcard de révision</h3>
              <h2 className="text-2xl font-bold mb-6">{card.title}</h2>
              
              <div className="bg-white/5 text-left p-6 rounded-xl border border-white/10 text-lg leading-relaxed text-slate-200">
                <LatexText text={card.content} />
              </div>

              <button onClick={advanceQuestion} className="mt-8 bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-full font-bold transition-all shadow-lg hover:shadow-blue-500/50 flex items-center gap-2 mx-auto active:scale-95">
                Compris <ArrowRight className="w-4 h-4" />
              </button>
            </div>
         </div>
       </div>
     );
   }

  // 5. QUIZ RUNNING
  const currentQ = quizQuestions[currentQIndex];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center pt-8 px-4 pb-20 font-sans">
      <div className="w-full max-w-2xl">
        {/* Top Bar */}
        <div className="flex items-center justify-between mb-6">
          <button onClick={() => setAppState('menu')} className="text-slate-400 hover:text-slate-600 transition-colors">
            <XCircle size={24} />
          </button>
          <div className="flex flex-col items-end">
             <span className="text-sm font-bold text-slate-700">Question {currentQIndex + 1} / {quizQuestions.length}</span>
             <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase mt-1 ${currentQ.diff === 'hard' ? 'bg-red-100 text-red-700' : 'bg-blue-50 text-blue-600'}`}>
               {currentQ.diff}
             </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-2 bg-slate-200 rounded-full mb-8 overflow-hidden">
          <div 
            className="h-full bg-blue-600 transition-all duration-300 ease-out" 
            style={{width: `${((currentQIndex + 1) / quizQuestions.length) * 100}%`}}
          ></div>
        </div>

        {/* Question Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden relative min-h-[400px] flex flex-col">
          <div className="p-8 flex-1">
            {/* Question Type Badge */}
            <div className="flex gap-2 mb-4">
              {currentQ.type === 'mcq' ? (
                <span className="inline-flex items-center gap-1 bg-indigo-50 text-indigo-700 text-xs font-bold px-2 py-1 rounded">
                   <List size={12} /> QCM
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 bg-amber-50 text-amber-700 text-xs font-bold px-2 py-1 rounded">
                   <MessageSquare size={12} /> Réponse Libre
                </span>
              )}
            </div>

            <h2 className="text-xl font-bold text-slate-800 mb-8 leading-relaxed">
              <LatexText text={currentQ.q} />
            </h2>

            {/* --- MCQ INTERFACE --- */}
            {currentQ.type === 'mcq' && (
              <div className="space-y-3">
                {currentQ.options.map((opt, idx) => {
                  let style = "w-full text-left p-4 rounded-xl border-2 transition-all flex items-center justify-between group relative overflow-hidden ";
                  if (selectedOption === null) {
                    style += "border-slate-100 hover:border-blue-200 hover:bg-blue-50/50 text-slate-700";
                  } else {
                    if (idx === currentQ.correct) style += "border-green-500 bg-green-50 text-green-900";
                    else if (idx === selectedOption) style += "border-red-500 bg-red-50 text-red-900";
                    else style += "border-slate-100 opacity-40";
                  }
                  
                  return (
                    <button 
                      key={idx} 
                      onClick={() => handleMCQOptionClick(idx)} 
                      disabled={selectedOption !== null} 
                      className={style}
                    >
                      <span className="font-medium relative z-10"><LatexText text={opt} /></span>
                      {selectedOption !== null && idx === currentQ.correct && <CheckCircle className="text-green-600 relative z-10" size={20} />}
                      {selectedOption !== null && idx === selectedOption && idx !== currentQ.correct && <XCircle className="text-red-600 relative z-10" size={20} />}
                    </button>
                  );
                })}
              </div>
            )}

            {/* --- OPEN QUESTION INTERFACE --- */}
            {currentQ.type === 'open' && (
              <div className="flex flex-col items-center justify-center py-4">
                {!isRevealed ? (
                  <div className="text-center w-full">
                    <p className="text-slate-500 italic mb-6">Réfléchissez à la réponse, puis cliquez pour vérifier.</p>
                    <textarea 
                      className="w-full p-4 border border-slate-200 rounded-xl bg-slate-50 mb-4 focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all"
                      placeholder="(Optionnel) Tapez votre idée ici..."
                      rows={3}
                    />
                    <button 
                      onClick={handleOpenReveal}
                      className="bg-blue-600 text-white px-8 py-3 rounded-full font-bold hover:bg-blue-700 transition-all shadow-lg flex items-center gap-2 mx-auto active:scale-95"
                    >
                      <Eye size={20} /> Révéler la réponse
                    </button>
                  </div>
                ) : (
                  <div className="w-full animate-in fade-in slide-in-from-bottom-4">
                    <div className="bg-slate-100 p-6 rounded-xl border-l-4 border-blue-500 mb-6">
                      <h3 className="text-xs font-bold text-slate-500 uppercase mb-2">Réponse attendue</h3>
                      <p className="text-lg font-bold text-slate-800 leading-relaxed"><LatexText text={currentQ.answer} /></p>
                    </div>

                    {!showExplanation && (
                      <div className="text-center">
                        <p className="text-slate-700 font-medium mb-4">Avez-vous eu juste ?</p>
                        <div className="flex justify-center gap-4">
                          <button 
                            onClick={() => handleOpenSelfEval(false)}
                            className="flex-1 py-3 bg-red-100 text-red-700 rounded-xl font-bold hover:bg-red-200 transition-colors active:scale-95"
                          >
                            Non 😞
                          </button>
                          <button 
                            onClick={() => handleOpenSelfEval(true)}
                            className="flex-1 py-3 bg-green-100 text-green-700 rounded-xl font-bold hover:bg-green-200 transition-colors active:scale-95"
                          >
                            Oui ! 🎯
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* EXPLANATION PANEL */}
          {showExplanation && (
            <div className={`p-6 border-t ${isCorrect ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'} animate-in slide-in-from-bottom-2`}>
              <div className="flex gap-3 mb-4">
                <div className={`mt-1 flex-shrink-0 ${isCorrect ? 'text-green-600' : 'text-red-600'}`}>
                  {isCorrect ? <CheckCircle size={24} /> : <XCircle size={24} />}
                </div>
                <div>
                  <h3 className={`font-bold ${isCorrect ? 'text-green-800' : 'text-red-800'}`}>
                    {isCorrect ? "Excellente réponse !" : "Pas tout à fait..."}
                  </h3>
                  <div className="text-slate-700 mt-2 leading-relaxed text-sm md:text-base">
                    <LatexText text={currentQ.expl} />
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end">
                <button 
                  onClick={handleNext}
                  className="bg-slate-900 text-white px-6 py-3 rounded-xl font-bold hover:bg-slate-800 transition-all flex items-center gap-2 shadow-lg active:scale-95"
                >
                  {currentQIndex < quizQuestions.length - 1 ? "Continuer" : "Voir les Résultats"} 
                  <ArrowRight size={18} />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Sub-component for Menu Button
function MenuButton({ icon, title, subtitle, onClick }) {
  return (
    <button 
      onClick={onClick}
      className="flex items-center gap-4 p-4 rounded-2xl border border-slate-200 bg-white hover:border-blue-400 hover:shadow-md transition-all text-left group w-full active:scale-[0.98]"
    >
      <div className="p-3 bg-slate-50 rounded-xl group-hover:bg-blue-50 transition-colors flex-shrink-0">
        {icon}
      </div>
      <div>
        <h3 className="font-bold text-slate-800 group-hover:text-blue-700 transition-colors">{title}</h3>
        <p className="text-xs text-slate-500">{subtitle}</p>
      </div>
      <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
        <ArrowRight className="w-5 h-5 text-blue-400" />
      </div>
    </button>
  );
}