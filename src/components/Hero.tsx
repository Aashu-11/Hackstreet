import React, { useEffect, useRef } from 'react';
import { ArrowRight } from 'lucide-react';

const Hero = () => {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const paragraphRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const animateElements = () => {
      if (headingRef.current) {
        headingRef.current.classList.add('animate-fade-in');
      }
      
      setTimeout(() => {
        if (paragraphRef.current) {
          paragraphRef.current.classList.add('animate-fade-in');
        }
      }, 400);
      
      setTimeout(() => {
        if (ctaRef.current) {
          ctaRef.current.classList.add('animate-fade-in');
        }
      }, 800);
    };

    // Start animations after a short delay
    setTimeout(animateElements, 100);
  }, []);

  return (
    <section className="min-h-screen flex items-center justify-center pt-20 pb-24 px-6">
      <div className="max-w-6xl mx-auto text-center">
        <h1 
          ref={headingRef}
          className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 opacity-0 transition-opacity duration-1000"
        >
          Discover Academic <span className="text-[#E4FD75]">Research</span> That Matters To You
        </h1>
        
        <p 
          ref={paragraphRef}
          className="text-lg md:text-xl max-w-3xl mx-auto mb-10 opacity-0 transition-opacity duration-1000 delay-200 text-gray-300"
        >
          Our AI-powered system analyzes your interests and research history to recommend
          relevant academic journals and papers tailored specifically to your needs.
        </p>
        
        <div 
          ref={ctaRef}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 opacity-0 transition-opacity duration-1000 delay-400"
        >
          <button 
            className="bg-[#E4FD75] text-[#282828] px-8 py-4 rounded-full font-medium hover:scale-105 transition-all duration-300 shadow-[0_0_20px_rgba(228,253,117,0.3)] group flex items-center gap-2"
            onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
          >
            Get Started
            <ArrowRight className="transition-transform duration-300 group-hover:translate-x-1" size={18} />
          </button>
        </div>
      </div>
    </section>
  );
};

export default Hero;