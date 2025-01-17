// VoiceLogger.js
import { useEffect } from 'react';

const VoiceLogger = ({ voices }) => {
  useEffect(() => {
    console.log('Available Voices:', voices);
    const samVoice = voices.find(voice => voice.name === 'Samantha');
    if (samVoice) {
      console.log('Samantha is available:', samVoice);
    } else {
      console.warn('Samantha voice is not available.');
    }
  }, [voices]);

  return null; // This component doesn't render anything visible
};

export default VoiceLogger;
