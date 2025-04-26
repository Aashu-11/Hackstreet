import React from 'react';
import { ArrowUpRight, Github, Twitter, Linkedin, BookOpenText } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="py-16 px-6 border-t border-white/10">
      <div className="max-w-7xl mx-auto">
        
        {/* Top section: About */}
        <div className="grid grid-cols-1 md:grid-cols-1 gap-12">
          
          {/* About */}
          <div className="md:col-span-1 w-full">
            <div className="flex items-center gap-2 mb-4">
              <BookOpenText size={32} className="text-[#E4FD75]" />
              <span className="text-2xl font-semibold">JournalSense</span>
            </div>

            <p className="text-gray-300 text-lg mb-6 leading-relaxed">
              JournalSense revolutionizes the way researchers discover and publish their work. Powered by cutting-edge AI, we intelligently map your research goals to the most impactful journals, saving you time and maximizing your reach. From citation networks to evolving research trends, JournalAI reads between the lines — so you can focus on making breakthroughs, not searching for them.
            </p>

            <div className="flex space-x-6 w-full">
              {[
                { href: '#', Icon: Twitter },
                { href: '#', Icon: Github },
                { href: '#', Icon: Linkedin },
              ].map(({ href, Icon }, idx) => (
                <a
                  key={idx}
                  href={href}
                  className="text-gray-400 hover:text-[#E4FD75] transition-all duration-300 transform hover:scale-110"
                >
                  <Icon size={24} />
                </a>
              ))}
            </div>
          </div>

        </div>

        {/* Bottom section: Legal */}
        <div className="mt-16 pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm mb-4 md:mb-0">
            © {new Date().getFullYear()} JournalAI. All rights reserved.
          </p>

          <div className="flex space-x-8">
            {[
              { label: 'Privacy Policy', href: '#' },
              { label: 'Terms of Service', href: '#' },
            ].map(({ label, href }, idx) => (
              <a
                key={idx}
                href={href}
                className="text-gray-400 hover:text-[#E4FD75] transition-colors duration-300 text-sm group"
              >
                <span className="relative">
                  {label}
                  <span className="absolute -bottom-1 left-0 w-full h-0.5 bg-[#E4FD75] transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
                </span>
              </a>
            ))}
          </div>
        </div>

      </div>
    </footer>
  );
};

export default Footer;
