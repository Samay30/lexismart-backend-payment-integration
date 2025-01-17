import React, { useState, useEffect, useRef } from 'react';


const SpeechRecognitionComponent = ({ onTranscript, handleButtonSpeech }) => {
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('Speech Recognition API not supported in this browser.');
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onstart = () => {
      setIsListening(true);
    };

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      onTranscript(transcript);
    };

    recognitionRef.current.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };
  }, [onTranscript]);

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setTimeout(() => {
        recognitionRef.current.start();
      }, 1000); // Delay by 1000 ms (1 second)
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  };

  const buttonStyle = {
    padding: '8px 12px',
    backgroundColor: '#5c91d9',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginRight: '10px',
  };

  return (
    <div style={{ marginTop: '20px', textAlign: 'center' }}>
      
      <button
        onClick={startListening}
        disabled={isListening}
        style={{
          ...buttonStyle,
          backgroundColor: isListening ? '#95a5a6' : '#3498db',
        }}
        onFocus={() => handleButtonSpeech('Start Listening')}
        onMouseEnter={() => handleButtonSpeech('Start Listening')}
      >
        {isListening ? 'Listening...' : 'Start Listening'}
      </button>
      <button
        onClick={stopListening}
        style={{
          ...buttonStyle,
          backgroundColor: '#e74c3c',
        }}
        onFocus={() => handleButtonSpeech('Stop Listening')}
        onMouseEnter={() => handleButtonSpeech('Stop Listening')}
      >
        Stop Listening
      </button>
      
      
    </div>
  );
};

export default SpeechRecognitionComponent;
