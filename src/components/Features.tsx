import React, { useEffect, useRef } from 'react';
import { Search, BookText, LineChart } from 'lucide-react';
import FeatureCard from './FeatureCard';
import AnimatedCircle from './AnimatedCircle';

const Features = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            if (entry.target === sectionRef.current) {
              titleRef.current?.classList.add('animate-title');
            }
          }
        });
      },
      { threshold: 0.2 }
    );
    
    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }
    
    return () => {
      if (sectionRef.current) {
        observer.unobserve(sectionRef.current);
      }
    };
  }, []);

  return (
    <section 
      id="features" 
      ref={sectionRef}
      className="min-h-screen py-24 px-6 relative opacity-0 transform translate-y-10 transition-all duration-1000 ease-out"
    >
      <div className="max-w-7xl mx-auto relative">
        <h2 
          ref={titleRef}
          className="text-4xl md:text-6xl font-bold text-center mb-20 opacity-0 transform translate-y-10 transition-all duration-1000 ease-out"
        >
          Powered by <span className="text-[#E4FD75] relative">
            Advanced AI
            <span className="absolute -bottom-2 left-0 w-full h-0.5 bg-[#E4FD75] transform scale-x-0 transition-transform duration-700 group-hover:scale-x-100"></span>
          </span>
        </h2>
        
        <div className="relative flex flex-col lg:flex-row items-center justify-center gap-8 lg:gap-16 perspective-1000">
  <div className="absolute opacity-40 w-full h-full pointer-events-none z-0 animate-pulse">
    <AnimatedCircle />
  </div>
  
  <div className="grid grid-cols-1 gap-12 z-10 w-full perspective-card items-center justify-center">
    <FeatureCard 
      icon={<BookText size={32} className="text-[#E4FD75]" />}
      title="Journal Suggestions"
      description="Advanced algorithms connect you with the most relevant academic publications from our vast database of peer-reviewed journals."
      delay={0}
    />
    
    <FeatureCard 
      icon={<LineChart size={32} className="text-[#E4FD75]" />}
      title="Trending Analysis"
      description="Discover emerging trends in academic publishing with interactive charts. Spot top-performing journals and rising stars to stay ahead in your field."
      delay={200}
    />
    
    <FeatureCard 
      icon={<Search size={32} className="text-[#E4FD75]" />}
      title="Smart Search"
      description="Our AI deeply analyzes research papers, understanding methodologies, findings, and citations to match your interests perfectly."
      delay={400}
    />
    
    <FeatureCard 
      icon={<BookText size={32} className="text-[#E4FD75]" />}
      title="MesmerizeAbstractBot"
      description="Generate compelling abstracts effortlessly by entering your keywords. Let our AI craft the perfect summary for your research."
      delay={600}
    />
  </div>

  <div className="mt-16 text-center z-10">
    <a 
      href="http://localhost:5174/" 
      target="_blank" 
      rel="noopener noreferrer" 
      className="text-lg font-semibold text-[#E4FD75] hover:underline"
    >
    </a>
  </div>
</div>
        
      </div>
    </section>
  );
};

export default Features;