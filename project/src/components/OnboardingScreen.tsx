import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare } from 'lucide-react';

interface OnboardingScreenProps {
  onComplete: () => void;
}

const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ onComplete }) => {
  const [showWelcome, setShowWelcome] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  
  useEffect(() => {
    const welcomeTimeout = setTimeout(() => setShowWelcome(true), 500);
    const instructionsTimeout = setTimeout(() => setShowInstructions(true), 2000);
    
    return () => {
      clearTimeout(welcomeTimeout);
      clearTimeout(instructionsTimeout);
    };
  }, []);
  
  return (
    <motion.div 
      className="flex flex-col items-center justify-center min-h-screen p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="w-full max-w-2xl">
        <div className="flex flex-col space-y-6">
          <AnimatePresence>
            {showWelcome && (
              <motion.div
                className="flex items-start gap-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  duration: 0.5,
                  ease: "easeOut"
                }}
              >
                <div className="bg-accent rounded-full p-3 text-background flex-shrink-0">
                  <MessageSquare size={24} />
                </div>
                <div className="bg-background-light p-4 md:p-6 rounded-lg rounded-tl-none flex-1">
                  <p className="text-xl font-medium">
                    👋 Welcome to <span className="text-accent font-bold">MesmerizeAbstractBot</span>!
                  </p>
                </div>
              </motion.div>
            )}
            
            {showInstructions && (
              <motion.div
                className="flex items-start gap-4 ml-12"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  duration: 0.5,
                  ease: "easeOut"
                }}
              >
                <div className="bg-background-light p-4 md:p-6 rounded-lg flex-1">
                  <p className="text-lg">
                    Enter 3–5 keywords for your research paper, and I'll craft a publication-ready abstract in seconds.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        <AnimatePresence>
          {showInstructions && (
            <motion.button
              className="mt-12 w-full py-3 rounded-lg neon-button morphing-button"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onComplete}
            >
              Continue
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default OnboardingScreen;