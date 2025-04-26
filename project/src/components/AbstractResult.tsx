import { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, RotateCcw } from 'lucide-react';
import { Abstract } from '../types';

interface AbstractResultProps {
  abstract: Abstract;
  keywords: string[];
  onReset: () => void;
}

const AbstractResult: React.FC<AbstractResultProps> = ({ abstract, keywords, onReset }) => {
  const [copied, setCopied] = useState(false);
  
  const copyToClipboard = () => {
    const fullText = `
BACKGROUND & MOTIVATION
${abstract.background}

METHODS
${abstract.methods}

RESULTS & FINDINGS
${abstract.results}

CONCLUSION & IMPACT
${abstract.conclusion}
    `;
    
    navigator.clipboard.writeText(fullText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  
  return (
    <motion.div 
      className="w-full max-w-3xl mx-auto p-4 md:p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-6 flex justify-between items-center">
        <motion.h2 
          className="text-2xl font-bold"
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Your Generated Abstract
        </motion.h2>
        
        <div className="flex gap-2">
          <motion.button
            className="p-2 rounded-full text-accent hover:bg-background-light transition-colors"
            onClick={copyToClipboard}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            {copied ? <Check size={20} /> : <Copy size={20} />}
          </motion.button>
          
          <motion.button
            className="p-2 rounded-full text-neutral-400 hover:text-neutral-100 hover:bg-background-light transition-colors"
            onClick={onReset}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
          >
            <RotateCcw size={20} />
          </motion.button>
        </div>
      </div>
      
      <motion.div 
        className="mb-4 flex flex-wrap gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {keywords.map((keyword, index) => (
          <span key={index} className="px-2 py-1 bg-background-light rounded-full text-sm">
            {keyword}
          </span>
        ))}
      </motion.div>
      
      <motion.div 
        className="bg-background-light rounded-lg p-6 shadow-lg backdrop-blur-sm"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <section className="mb-4">
          <h3 className="section-title mb-2">BACKGROUND & MOTIVATION</h3>
          <p className="text-neutral-100 leading-relaxed">{abstract.background}</p>
        </section>
        
        <section className="mb-4">
          <h3 className="section-title mb-2">METHODS</h3>
          <p className="text-neutral-100 leading-relaxed">{abstract.methods}</p>
        </section>
        
        <section className="mb-4">
          <h3 className="section-title mb-2">RESULTS & FINDINGS</h3>
          <p className="text-neutral-100 leading-relaxed">{abstract.results}</p>
        </section>
        
        <section>
          <h3 className="section-title mb-2">CONCLUSION & IMPACT</h3>
          <p className="text-neutral-100 leading-relaxed">{abstract.conclusion}</p>
        </section>
      </motion.div>
      
      <motion.button
        className="mt-8 w-full py-3 rounded-lg neon-button morphing-button"
        onClick={onReset}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        Generate Another Abstract
      </motion.button>
    </motion.div>
  );
};

export default AbstractResult;