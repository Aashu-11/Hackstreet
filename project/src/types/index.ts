export enum AppState {
  SPLASH = 'splash',
  ONBOARDING = 'onboarding',
  INPUT = 'input',
  LOADING = 'loading',
  RESULT = 'result',
}

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

export interface Abstract {
  background: string;
  methods: string;
  results: string;
  conclusion: string;
}