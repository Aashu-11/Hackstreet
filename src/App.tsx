import React from 'react';
import Hero from './components/Hero';
import Features from './components/Features';
import Footer from './components/Footer';
import Navbar from './components/Navbar';
import BackgroundAnimation from './components/BackgroundAnimation';

function App() {
  return (
    <div className="min-h-screen bg-[#282828] text-white overflow-hidden relative">
      <BackgroundAnimation />
      <Navbar />
      <main>
        <Hero />
        <Features />
      </main>
      <Footer />
    </div>
  );
}

export default App;