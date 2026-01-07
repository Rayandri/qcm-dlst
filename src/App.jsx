import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  BookOpen, CheckCircle, XCircle, Brain, ArrowRight, RotateCcw, 
  Award, Zap, Calculator, Activity, Shuffle, List, AlertTriangle, 
  Layers, Eye, MessageSquare, ChevronDown, ChevronUp, FileText, 
  Thermometer, Anchor, Network
} from 'lucide-react';
import { Analytics } from '@vercel/analytics/react';
import { database, flashcards, slideSummaries } from './data';

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
  const [questionCount, setQuestionCount] = useState('all'); // 10, 20, 30, 40, all

  // START QUIZ
  const startQuiz = (selectedMode) => {
    setMode(selectedMode);
    // 1. Deep Clone + Shuffle Options
    let q = database.map(item => {
      if (item.type !== 'mcq') return { ...item };
      
      // Shuffle options but track correct answer
      const optionsWithIndex = item.options.map((opt, idx) => ({ opt, idx }));
      const shuffled = shuffleArray(optionsWithIndex);
      
      return {
        ...item,
        options: shuffled.map(o => o.opt),
        correct: shuffled.findIndex(o => o.idx === item.correct)
      };
    });
    
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

    // Limiter le nombre de questions
    if (questionCount !== 'all') {
      q = q.slice(0, parseInt(questionCount));
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

          <div className="mb-8">
             <p className="text-sm font-bold text-slate-400 mb-3 uppercase tracking-wide">Nombre de questions</p>
             <div className="flex gap-2 justify-center">
               {['10', '20', '30', '40', 'all'].map(count => (
                 <button
                   key={count}
                   onClick={() => setQuestionCount(count)}
                   className={`px-4 py-2 rounded-lg font-bold transition-all ${questionCount === count 
                     ? 'bg-blue-600 text-white shadow-lg scale-105' 
                     : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
                 >
                   {count === 'all' ? 'Tout' : count}
                 </button>
               ))}
             </div>
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
              {currentQ.isBonus && (
                <span className="inline-flex items-center gap-1 bg-purple-50 text-purple-700 text-xs font-bold px-2 py-1 rounded">
                   ✨ Culture G
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
      <Analytics />
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