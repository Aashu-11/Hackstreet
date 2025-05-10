import React, { useEffect, useRef, ReactNode } from 'react';

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  delay: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description, delay }) => {
  const cardRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setTimeout(() => {
              if (cardRef.current) {
                cardRef.current.classList.add('is-visible');
              }
            }, delay);
          }
        });
      },
      { threshold: 0.2 }
    );
    
    if (cardRef.current) {
      observer.observe(cardRef.current);
    }
    
    return () => {
      if (cardRef.current) {
        observer.unobserve(cardRef.current);
      }
    };
  }, [delay]);

  return (
    <div 
      ref={cardRef}
      className="feature-card bg-[#1a1a1a]/70 backdrop-blur-sm rounded-2xl p-8 border border-white/10 shadow-lg opacity-0 transform translate-y-10 transition-all duration-700 ease-out group hover:border-[#E4FD75]/30"
    >
      <div className="bg-[#282828] rounded-xl p-4 inline-block mb-6 group-hover:bg-[#E4FD75]/10 transition-all duration-300 transform group-hover:scale-110">
        {icon}
      </div>
      
      <h3 className="text-2xl font-semibold mb-4 group-hover:text-[#E4FD75] transition-colors duration-300">
        {title}
      </h3>
      
      <p className="text-gray-400 leading-relaxed group-hover:text-gray-300 transition-colors duration-300">
        {description}
      </p>
      
      <button 
        className="mt-6 text-[#E4FD75] flex items-center gap-2 font-medium group-hover:translate-x-2 transition-transform duration-300"
        onClick={() => {
          if (title === "Journal Suggestions") {
            window.location.href = "https://corefunctionality.streamlit.app/";
          } else if (title === "Trending Analysis") {
            window.location.href = "https://dashboardforanalysis.streamlit.app/";
          }else if(title === "Smart Search"){
            window.location.href = "https://keywordfinder.streamlit.app/";
          }else if(title === "MesmerizeAbstractBot"){
            window.location.href = "https://journalsensebot.vercel.app/";
          }
        }}
      >
        Try Demo
        <span className="transform group-hover:translate-x-1 transition-transform duration-300">→</span>
      </button>
    </div>
  );
};

export default FeatureCard;
