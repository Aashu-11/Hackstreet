/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#282828',
        'background-light': '#363636',
        accent: '#E4FD75',
        'accent-dark': '#C8E25E',
        neutral: {
          100: '#FFFFFF',
          200: '#F2F2F2',
          300: '#E0E0E0',
          400: '#BDBDBD',
          500: '#9E9E9E',
          600: '#757575',
          700: '#616161',
          800: '#424242',
          900: '#212121',
        },
      },
      fontFamily: {
        sans: ['Space Grotesk', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 3s infinite',
        'typing': 'typing 1.5s steps(20) infinite alternate, blink .7s infinite',
      },
      keyframes: {
        typing: {
          '0%': {
            width: '0%',
            visibility: 'hidden'
          },
          '100%': {
            width: '100%'
          }
        },
        blink: {
          '50%': {
            borderColor: 'transparent'
          },
          '100%': {
            borderColor: 'white'
          }
        }
      },
      boxShadow: {
        'neon': '0 0 5px theme(colors.accent), 0 0 20px theme(colors.accent)',
        'neon-sm': '0 0 2px theme(colors.accent), 0 0 10px theme(colors.accent)',
      },
    },
  },
  plugins: [],
};