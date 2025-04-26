import { motion } from 'framer-motion';

const LoadingScreen: React.FC = () => {
  return (
    <motion.div 
      className="flex flex-col items-center justify-center min-h-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="relative">
        <motion.div 
          className="w-24 h-24 rounded-full border-4 border-neutral-700"
          initial={{ opacity: 0.3 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, repeat: Infinity, repeatType: "reverse" }}
        />
        
        <motion.div
          className="absolute inset-0 w-24 h-24 rounded-full border-t-4 border-accent"
          animate={{ 
            rotate: 360,
          }}
          transition={{ 
            duration: 1.5, 
            repeat: Infinity,
            ease: "linear" 
          }}
        />
        
        {[0, 1, 2].map((index) => (
          <motion.div
            key={index}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-accent rounded-full"
            initial={{ 
              scale: 0.5,
              opacity: 0.7,
            }}
            animate={{ 
              scale: [0.5, 1.5, 0.5],
              opacity: [0.7, 0, 0.7],
            }}
            transition={{ 
              duration: 2,
              delay: index * 0.5,
              repeat: Infinity,
            }}
            style={{
              width: `${8 + index * 4}px`,
              height: `${8 + index * 4}px`,
            }}
          />
        ))}
      </div>
      
      <motion.p 
        className="mt-8 text-lg text-neutral-300"
        animate={{ 
          opacity: [0.5, 1, 0.5] 
        }}
        transition={{ 
          duration: 2, 
          repeat: Infinity, 
        }}
      >
        Generating your abstract...
      </motion.p>
    </motion.div>
  );
};

export default LoadingScreen;