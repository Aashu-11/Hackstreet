import { motion } from 'framer-motion';
import { Code } from 'lucide-react';

interface SplashScreenProps {
  onComplete: () => void;
}

const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete }) => {
  return (
    <motion.div 
      className="flex flex-col items-center justify-center h-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="flex flex-col items-center gap-6"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ 
          delay: 0.3, 
          duration: 0.8,
          ease: [0.16, 1, 0.3, 1]
        }}
      >
        <motion.div 
          className="text-accent p-4 rounded-full border-2 border-accent shadow-neon"
          animate={{ 
            boxShadow: [
              '0 0 5px #E4FD75, 0 0 20px #E4FD75',
              '0 0 10px #E4FD75, 0 0 30px #E4FD75',
              '0 0 5px #E4FD75, 0 0 20px #E4FD75',
            ] 
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity,
            repeatType: 'reverse'
          }}
        >
          <Code size={64} />
        </motion.div>
        
        <motion.h1 
          className="text-4xl md:text-5xl font-bold text-center"
          animate={{ 
            textShadow: [
              '0 0 5px rgba(228, 253, 117, 0.5)',
              '0 0 10px rgba(228, 253, 117, 0.8)',
              '0 0 5px rgba(228, 253, 117, 0.5)',
            ] 
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity,
            repeatType: 'reverse'
          }}
        >
          <span className="text-accent">Mesmerize</span>AbstractBot
        </motion.h1>
      </motion.div>
      
      <motion.button
        className="mt-16 px-6 py-3 rounded-full neon-button morphing-button"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5, duration: 0.5 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onComplete}
      >
        Get Started
      </motion.button>
    </motion.div>
  );
};

export default SplashScreen;