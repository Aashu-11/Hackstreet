import React, { useState, useEffect } from 'react';
import { BookOpenText, Menu, X } from 'lucide-react';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav 
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ease-in-out px-6 py-4 ${
        isScrolled ? 'bg-[#282828]/90 backdrop-blur-md' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <a href="#" className="flex items-center gap-2 group">
          <BookOpenText 
            size={32} 
            className="text-[#E4FD75] transition-transform duration-300 group-hover:rotate-12" 
          />
          <span className="text-xl font-semibold">JournalSense</span>
        </a>
        
        {/* Desktop menu */}
        <div className="hidden md:flex items-center gap-8">
          <a href="#features" className="hover:text-[#E4FD75] transition-colors duration-200">Features</a>
        </div>
        
        {/* Mobile menu button */}
        <button 
          className="md:hidden text-white"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
      
      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-[#1e1e1e] absolute top-full left-0 w-full py-4 px-6 flex flex-col gap-4 shadow-lg">
          <a 
            href="#features" 
            className="hover:text-[#E4FD75] transition-colors duration-200 py-2"
            onClick={() => setIsMenuOpen(false)}
          >
            Features
          </a>
        </div>
      )}
    </nav>
  );
};

export default Navbar;