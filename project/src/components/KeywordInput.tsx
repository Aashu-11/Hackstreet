import { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, AlertCircle } from 'lucide-react';

interface KeywordInputProps {
  onSubmit: (keywords: string[]) => void;
}

const KeywordInput: React.FC<KeywordInputProps> = ({ onSubmit }) => {
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const keywords = input
      .split(',')
      .map(keyword => keyword.trim())
      .filter(keyword => keyword.length > 0);
    
    if (keywords.length < 3) {
      setError('Please enter at least 3 keywords separated by commas');
      return;
    }
    
    setError(null);
    onSubmit(keywords);
  };
  
  return (
    <motion.div 
      className="w-full max-w-2xl mx-auto p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold mb-2">Enter Your Keywords</h2>
        <p className="text-neutral-300">
          Provide 3-5 keywords separated by commas to generate your abstract
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="w-full">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="machine learning, neural networks, computer vision..."
            className="w-full input-field pr-12 py-4"
            autoFocus
          />
          <motion.button
            type="submit"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-accent p-2 rounded-full hover:bg-background-light transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Send size={20} />
          </motion.button>
        </div>
        
        {error && (
          <motion.div 
            className="flex items-center mt-3 text-accent"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AlertCircle size={16} className="mr-2" />
            <p className="text-sm">{error}</p>
          </motion.div>
        )}
        
        <div className="mt-8">
          <h3 className="text-lg font-medium text-center mb-4">Example Keywords</h3>
          <div className="flex flex-wrap justify-center gap-2">
            {[
              'quantum computing, cryptography, information theory',
              'climate change, renewable energy, sustainability',
              'gene editing, CRISPR, bioethics',
              'artificial intelligence, deep learning, robotics'
            ].map((example, index) => (
              <motion.button
                key={index}
                type="button"
                className="px-3 py-1 text-sm bg-background-light rounded-full hover:text-accent hover:border-accent transition-colors border border-neutral-700"
                onClick={() => setInput(example)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {example}
              </motion.button>
            ))}
          </div>
        </div>
      </form>
    </motion.div>
  );
};

export default KeywordInput;