import React, { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { AppState, Abstract } from './types';
import ParticleBackground from './components/ParticleBackground';
import SplashScreen from './components/SplashScreen';
import OnboardingScreen from './components/OnboardingScreen';
import KeywordInput from './components/KeywordInput';
import LoadingScreen from './components/LoadingScreen';
import AbstractResult from './components/AbstractResult';
import { generateAbstract } from './services/gemini';

function App() {
  const [appState, setAppState] = useState<AppState>(AppState.SPLASH);
  const [keywords, setKeywords] = useState<string[]>([]);
  const [abstract, setAbstract] = useState<Abstract | null>(null);

  const handleSplashComplete = () => {
    setAppState(AppState.ONBOARDING);
  };

  const handleOnboardingComplete = () => {
    setAppState(AppState.INPUT);
  };

  const handleKeywordSubmit = async (submittedKeywords: string[]) => {
    setKeywords(submittedKeywords);
    setAppState(AppState.LOADING);

    try {
      const generatedAbstract = await generateAbstract(submittedKeywords);
      setAbstract(generatedAbstract);
      setAppState(AppState.RESULT);
    } catch (error) {
      console.error('Error generating abstract:', error);
      // In a real app, we would handle this error more gracefully
      setAppState(AppState.INPUT);
    }
  };

  const handleReset = () => {
    setAppState(AppState.INPUT);
    setKeywords([]);
    setAbstract(null);
  };

  return (
    <div className="min-h-screen w-full relative overflow-hidden">
      <ParticleBackground />
      
      <AnimatePresence mode="wait">
        {appState === AppState.SPLASH && (
          <SplashScreen onComplete={handleSplashComplete} />
        )}
        
        {appState === AppState.ONBOARDING && (
          <OnboardingScreen onComplete={handleOnboardingComplete} />
        )}
        
        {appState === AppState.INPUT && (
          <KeywordInput onSubmit={handleKeywordSubmit} />
        )}
        
        {appState === AppState.LOADING && (
          <LoadingScreen />
        )}
        
        {appState === AppState.RESULT && abstract && (
          <AbstractResult 
            abstract={abstract} 
            keywords={keywords}
            onReset={handleReset}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;