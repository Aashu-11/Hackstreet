import React, { useEffect, useRef } from 'react';

const AnimatedCircle = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    const setDimensions = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    setDimensions();
    window.addEventListener('resize', setDimensions);
    
    // Animation parameters
    let time = 0;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;
    
    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw the main circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      
      // Create gradient
      const gradient = ctx.createLinearGradient(
        centerX - radius, centerY, centerX + radius, centerY
      );
      gradient.addColorStop(0, 'rgba(228, 253, 117, 0.1)');
      gradient.addColorStop(0.5, 'rgba(228, 253, 117, 0.3)');
      gradient.addColorStop(1, 'rgba(228, 253, 117, 0.1)');
      
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.restore();
      
      // Draw orbiting particles
      const numParticles = 24;
      
      for (let i = 0; i < numParticles; i++) {
        const angle = (i / numParticles) * Math.PI * 2 + time;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        const size = 2 + Math.sin(time * 2 + i) * 1.5;
        
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(228, 253, 117, 0.7)';
        ctx.fill();
        ctx.restore();
      }
      
      // Draw connecting lines
      ctx.save();
      for (let i = 0; i < numParticles; i++) {
        const angle1 = (i / numParticles) * Math.PI * 2 + time;
        const x1 = centerX + Math.cos(angle1) * radius;
        const y1 = centerY + Math.sin(angle1) * radius;
        
        for (let j = i + 1; j < numParticles; j += 3) {
          const angle2 = (j / numParticles) * Math.PI * 2 + time;
          const x2 = centerX + Math.cos(angle2) * radius;
          const y2 = centerY + Math.sin(angle2) * radius;
          
          const distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
          
          if (distance < radius) {
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = `rgba(228, 253, 117, ${0.1 - distance / (radius * 10)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      ctx.restore();
      
      // Inner pulsing circle
      ctx.save();
      ctx.beginPath();
      const pulseRadius = radius * 0.6 + Math.sin(time * 2) * radius * 0.05;
      ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(228, 253, 117, 0.05)';
      ctx.fill();
      ctx.restore();
      
      // Increment time
      time += 0.005;
      
      // Request next frame
      requestAnimationFrame(animate);
    };
    
    // Start animation
    animate();
    
    return () => {
      window.removeEventListener('resize', setDimensions);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute top-0 left-0 w-full h-full"
    />
  );
};

export default AnimatedCircle;