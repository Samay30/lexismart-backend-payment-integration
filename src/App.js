// App.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './App.css';
import ReactFlow, {
  addEdge,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'react-flow-renderer';
import styled, { ThemeProvider } from 'styled-components';
import { FiPlus, FiTrash2, FiSave, FiDownload, FiSun, FiMoon } from 'react-icons/fi';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import VoiceLogger from './VoiceLogger';
import SpeechRecognitionComponent from './SpeechRecognitionComponent';
import { FaFacebook, FaTwitter, FaLinkedin } from 'react-icons/fa';
import {
  Header,
  AppContainer,
  Card,
  ButtonContainer,
  Button,
  TextArea,
  InputField,
  SelectField,
  MindMapContainer,
  SummaryHeader,
  SubHeader,
  StyledLabel,
  FeedbackMessage,
  Header2,
  Header3,
  Footer
} from './StyledComponents'; // Ensure correct path
import { lightTheme, darkTheme } from './themes'; // Ensure correct path

// -------------------- Custom Node Component --------------------
const customNodeStyle = {
  padding: '10px',
  borderRadius: '5px',
  backgroundColor: '#FFCC00',
  color: '#333',
  border: '2px solid #fff',
  boxShadow: '0px 1px 3px rgba(0,0,0,0.2)',
  minWidth: '100px',
  textAlign: 'center',
  cursor: 'pointer',
};

const CustomNode = ({ data }) => {
  return (
    <div style={customNodeStyle} aria-label={`Concept: ${data.label}`}>
      {data.label}
    </div>
  );
};

// -------------------- Define Node Types --------------------
const nodeTypes = {
  custom: CustomNode,
};

// -------------------- Initial Nodes --------------------
const initialNodes = [
  {
    id: '1',
    type: 'input',
    data: { label: 'Central Idea' },
    position: { x: 250, y: 5 },
  },
];

function App() {
  // -------------------- Theme State --------------------
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme ? savedTheme : 'light';
  });

  // Toggle Theme Function
  const toggleTheme = () => {
    setTheme((prev) => {
      const newTheme = prev === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', newTheme);
      return newTheme;
    });
  };

  // -------------------- State Variables --------------------
  // Text Summarization
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const summarizeText = async () => {
    if (!inputText.trim()) {
      setSummary('Please enter some text to summarize.');
      return;
    }
    try {
      const response = await axios.post('/summarize', { text: inputText }); // Adjust the endpoint as needed
      const summaryData = response.data;
      if (summaryData && summaryData[0] && summaryData[0].summary_text) {
        setSummary(summaryData[0].summary_text);
      } else {
        setSummary('No summary available.');
      }
    } catch (error) {
      console.error('Error in summarizing:', error);
      setSummary('Failed to generate summary.');
    }
  };


  // OCR
  const [ocrText, setOcrText] = useState('');

  // Speech Synthesis
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [voices, setVoices] = useState([]);
  const [speechRate, setSpeechRate] = useState(1); // Default rate
  const [highlightedWordIndex, setHighlightedWordIndex] = useState(-1);

  // Question and Answer Builder
  const [questionsList, setQuestionsList] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [userAnswer, setUserAnswer] = useState('');
  const [selectedQuestion, setSelectedQuestion] = useState('');
  const [feedback, setFeedback] = useState('');
  const [isCorrect, setIsCorrect] = useState(null);

  // Mind Mapping
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [speechError, setSpeechError] = useState('');

  // -------------------- Prevent Welcome Message from Repeating --------------------
  const hasSaidWelcome = useRef(false);

  // -------------------- Voice Selection and Speech Synthesis --------------------
  useEffect(() => {
    const synth = window.speechSynthesis;

    const loadVoices = () => {
      let availableVoices = synth.getVoices();

      if (availableVoices.length !== 0) {
        setVoices(availableVoices);

        // Filter for en-US voices
        const enUSVoices = availableVoices.filter(voice => voice.lang === 'en-US');

        let preferredVoice = null;

        // Detect browser (Safari vs. others)
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

        if (isSafari) {
          // Safari-specific voices
          preferredVoice = enUSVoices.find(voice => 
            ['Samantha'].includes(voice.name)
          );
        } else {
          // Chrome and other browsers
          preferredVoice = enUSVoices.find(voice => voice.name === 'Google US English');
        }

        // Fallback to first en-US voice if preferred not found
        if (!preferredVoice && enUSVoices.length > 0) {
          preferredVoice = enUSVoices[0];
        }

        // Fallback to default voice if no en-US voices are found
        if (!preferredVoice && availableVoices.length > 0) {
          preferredVoice = availableVoices[0];
        }

        if (preferredVoice) {
          setSelectedVoice(preferredVoice.name);
          console.log(`Selected Voice: ${preferredVoice.name}`); // Confirm selection
        }

        // Check if speech synthesis is supported
        if (synth) {
          setSpeechSupported(true);
        }
      } else {
        // Retry after voices are loaded
        setTimeout(loadVoices, 100);
      }
    };

    loadVoices();

    // Listen for voices changed event
    synth.onvoiceschanged = loadVoices;
  }, []);

  // Automatic Welcome Speech on App Load (Only once)
  useEffect(() => {
    if (voices.length === 0 || !selectedVoice || hasSaidWelcome.current) return; // Wait until voices are loaded and not yet said

    const welcomeMessage = 'Welcome to LexiSmart, a space where you can learn effectively.';

    const utterance = new SpeechSynthesisUtterance(welcomeMessage);
    const selectedVoiceObj = voices.find((voice) => voice.name === selectedVoice);
    if (selectedVoiceObj) {
      utterance.voice = selectedVoiceObj;
    }
    utterance.rate = speechRate;
    utterance.pitch = 1; // Default pitch
    utterance.lang = 'en-US'; // Ensure American English

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      hasSaidWelcome.current = true; // Mark as said
    };

    window.speechSynthesis.speak(utterance);
  }, [voices, selectedVoice, speechRate]);

  // Function to handle button navigation speech
  const handleButtonSpeech = (text) => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
    }

    const utterance = new SpeechSynthesisUtterance(text);
    const selectedVoiceObj = voices.find((voice) => voice.name === selectedVoice);
    if (selectedVoiceObj) {
      utterance.voice = selectedVoiceObj;
    }
    utterance.rate = speechRate;
    utterance.pitch = 1; // Default pitch
    utterance.lang = 'en-US'; // Ensure American English

    window.speechSynthesis.speak(utterance);
  };

  // Function to play synthesized speech
  const speakText = () => {
    if (isSpeaking) return;

    if (!summary) {
      alert('No text to speak.');
      return;
    }

    const utterance = new SpeechSynthesisUtterance(summary);
    const selectedVoiceObj = voices.find((voice) => voice.name === selectedVoice);

    if (selectedVoiceObj) {
      utterance.voice = selectedVoiceObj;
    } else {
      // Fallback to first available en-US voice
      const fallbackVoice = voices.find((voice) => voice.lang === 'en-US');
      if (fallbackVoice) {
        utterance.voice = fallbackVoice;
        console.warn('Selected voice not found. Falling back to:', fallbackVoice.name);
      }
    }

    utterance.rate = speechRate; // Ensure this is within a reasonable range, e.g., 0.8 - 1.2
    utterance.pitch = 1; // Default pitch
    utterance.lang = 'en-US'; // Ensure American English

    const words = summary.split(' ');
    let wordIndex = 0;

    // Highlight words as they are spoken
    utterance.onboundary = (event) => {
      if (event.name === 'word') {
        setHighlightedWordIndex(wordIndex);
        wordIndex++;
      }
    };

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setHighlightedWordIndex(-1);
    };

    window.speechSynthesis.speak(utterance);
  };

  // Stop the speech
  const stopSpeech = () => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      setHighlightedWordIndex(-1);
    }
  };

  // -------------------- Question and Answer Builder Functions --------------------
  // Fetch questions on component mount
  useEffect(() => {
    axios
      .get('/get_questions') // Adjust the endpoint as needed
      .then((response) => setQuestionsList(response.data))
      .catch((error) => console.error('Error fetching questions:', error));
  }, []);

  // Handle form submission to add a new question
  const handleAddQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim() || !answer.trim()) {
      handleButtonSpeech('Please add a Question and an Answer');
      return;
    }
    try {
      const response = await axios.post('/add_question', { question, answer }); // Adjust the endpoint as needed
      const updatedQuestions = await axios.get('/get_questions');
      setQuestionsList(updatedQuestions.data);
      setQuestion('');
      setAnswer('');
    } catch (error) {
      console.error('Error adding question:', error);
      
    }
  };

  // Handle checking the user's answer
  // Handle checking the user's answer
const handleCheckAnswer = async (e) => {
  e.preventDefault();
  if (!selectedQuestion || !userAnswer.trim()) {
    handleButtonSpeech('Select a question and provide an answer.');
    return;
  }
  try {
    const response = await axios.post('/check_answer', { question: selectedQuestion, answer: userAnswer }); // Adjust the endpoint as needed
    setIsCorrect(response.data.correct);

    // Provide encouraging feedback
    if (response.data.correct) {
      handleButtonSpeech('You are correct! Great job!');
    } else {
      handleButtonSpeech('Almost there! Try again.');
    }

    setUserAnswer('');
  } catch (error) {
    console.error('Error checking answer:', error);
  }
};


  // -------------------- OCR Image Upload Functions --------------------
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    try {
      const response = await axios.post('/ocr', formData, { headers: { 'Content-Type': 'multipart/form-data' } }); // Adjust the endpoint as needed
      setOcrText(response.data.ocr_text);
    } catch (error) {
      console.error('There was an error with the OCR request:', error);
      setOcrText('Failed to extract text from image.');
    }
  };

  // -------------------- Mind Mapping Functions --------------------
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onElementsRemoveHandler = useCallback(
    (elementsToRemove) => {
      elementsToRemove.forEach((el) => {
        setNodes((nds) => nds.filter((node) => node.id !== el.id));
        setEdges((eds) => eds.filter((edge) => edge.id !== el.id));
      });
    },
    [setNodes, setEdges]
  );

  const onLoad = useCallback((reactFlowInstance) => {
    reactFlowInstance.fitView();
    setReactFlowInstance(reactFlowInstance);
  }, []);

  const addNodeHandler = () => {
    const newNode = {
      id: `${+new Date()}`,
      type: 'custom',
      data: { label: 'New Concept' },
      position: {
        x: Math.random() * 250 + 100,
        y: Math.random() * 250 + 100,
      },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const removeSelectedElementsHandler = () => {
    if (reactFlowInstance) {
      const selectedElements = reactFlowInstance.getElements().filter((el) => el.selected);
      if (selectedElements.length === 0) {
        alert('No elements selected.');
        return;
      }
      onElementsRemoveHandler(selectedElements);
    }
  };

  const saveMindMap = () => {
    const mindMapData = {
      nodes,
      edges,
    };
    localStorage.setItem('mindMap', JSON.stringify(mindMapData));
    alert('Mind map saved!');
  };

  const exportAsImage = () => {
    const mindMapElement = document.getElementById('mind-map-container');
    html2canvas(mindMapElement).then((canvas) => {
      const imgData = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = imgData;
      link.download = 'mindmap.png';
      link.click();
    });
  };

  const exportAsPDF = () => {
    const mindMapElement = document.getElementById('mind-map-container');
    html2canvas(mindMapElement).then((canvas) => {
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('landscape');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('mindmap.pdf');
    });
  };

  const onNodeClick = (event, node) => {
    setSelectedNodeId(node.id);
    handleButtonSpeech('Selected node. Click Start Listening to update label via voice.');
  };

  // -------------------- Speech Recognition for Mind Map Label Update --------------------
  const SpeechRecognitionAPI =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  const recognition = SpeechRecognitionAPI ? new SpeechRecognitionAPI() : null;

  useEffect(() => {
    if (!recognition) {
      console.warn('Speech Recognition API not supported in this browser.');
      setSpeechError('Speech Recognition is not supported in your browser.');
      return;
    }

    setSpeechSupported(true);

    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setFeedback('Listening...');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      if (selectedNodeId) {
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === selectedNodeId) {
              return {
                ...node,
                data: { ...node.data, label: transcript },
              };
            }
            return node;
          })
        );
        setFeedback(`Node updated to: "${transcript}"`);
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'no-speech') {
        setFeedback('No speech detected. Please try again.');
      } else if (event.error === 'audio-capture') {
        setFeedback('No microphone detected. Please connect a microphone.');
      } else if (event.error === 'not-allowed') {
        setFeedback('Microphone access denied. Please allow access.');
      } else {
        setFeedback(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
    };
  }, [recognition, selectedNodeId, setNodes]);

  const startListening = () => {
    if (!speechSupported) {
      alert('Speech Recognition not supported in this browser.');
      return;
    }
    try {
      recognition.start();
      setFeedback('Listening... Please speak.');
    } catch (error) {
      console.error('Error starting recognition:', error);
      setFeedback('Error starting speech recognition.');
    }
  };

  const stopListening = () => {
    if (recognition && isListening) {
      recognition.stop();
      setFeedback('Stopped listening.');
    }
  };

  // -------------------- Render Highlighted Summary Text --------------------
  const renderHighlightedSummary = () => {
    if (!summary) return null;
    return (
      <SummaryHeader>
        {summary.split(' ').map((word, index) => (
          <span
            key={index}
            id={`word-${index}`}
            className={highlightedWordIndex === index ? 'highlight' : ''}
          >
            {word + ' '}
          </span>
        ))}
      </SummaryHeader>
    );
  };

  return (
    <ThemeProvider theme={theme === 'light' ? lightTheme : darkTheme}>
      <AppContainer>
        <Header>LexiSmart</Header>

        {/* Dark Mode Toggle Button */}
        <ButtonContainer>
          <Button onClick={toggleTheme} bgColor={theme === 'light' ? '#2c3e50' : '#f39c12'}>
            {theme === 'light' ? <FiMoon /> : <FiSun />} {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
          </Button>
        </ButtonContainer>

        {/* Voice Logger for Debugging */}
        <VoiceLogger voices={voices} />

        {/* -------------------- Text Summarization Section -------------------- */}
        <Card>
          <Header2>Text Summarization</Header2>
          <TextArea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to summarize or speak"
          />
          <ButtonContainer>
            <Button
              onClick={summarizeText}
              onFocus={() => handleButtonSpeech('Summarize Text')}
              onMouseEnter={() => handleButtonSpeech('Summarize Text')}
            >
              Summarize Text
            </Button>
            
            <Button
              onClick={speakText}
              disabled={isSpeaking}
              bgColor={isSpeaking ? '#95a5a6' : '#3498db'}
              onFocus={() => handleButtonSpeech(isSpeaking ? 'Speaking...' : 'Speak Text')}
              onMouseEnter={() => handleButtonSpeech(isSpeaking ? 'Speaking...' : 'Speak Text')}
            >
              {isSpeaking ? 'Speaking...' : 'Speak Text'}
            </Button>
            <Button
              onClick={stopSpeech}
              disabled={!isSpeaking}
              bgColor={!isSpeaking ? '#95a5a6' : '#e74c3c'}
              onFocus={() => handleButtonSpeech('Stop Speech')}
              onMouseEnter={() => handleButtonSpeech('Stop Speech')}
            >
              Stop Speech
            </Button>
          </ButtonContainer>
          <div className="voice-selection">
            <StyledLabel htmlFor="voiceSelect">Select Voice:</StyledLabel>
            <SelectField
              id="voiceSelect"
              value={selectedVoice}
              onChange={(e) => {
                setSelectedVoice(e.target.value);
                localStorage.setItem('selectedVoice', e.target.value);
              }}
              aria-label="Select Voice"
            >
              {voices
                .filter((voice) => voice.lang === 'en-US') // Filter for US English voices
                .map((voice, index) => (
                  <option key={index} value={voice.name}>
                    {voice.name} ({voice.lang})
                  </option>
                ))}
            </SelectField>
          </div>

          <div className="rate-selection">
            <StyledLabel htmlFor="rateRange">Speech Rate:</StyledLabel>
            <input
              type="range"
              id="rateRange"
              min="0.8"
              max="1.2"
              step="0.1"
              value={speechRate}
              onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
              aria-label="Speech Rate"
            />
            <span>{speechRate.toFixed(1)}</span>
          </div>

          {renderHighlightedSummary()}
        </Card>


        {/* -------------------- Question and Answer Builder Section -------------------- */}
        <Card>
          <Header2>Question and Answer Builder</Header2>
          <form onSubmit={handleAddQuestion}>
            <div>
              <StyledLabel htmlFor="question">Enter a question:</StyledLabel>
              <InputField
                id="question"
                name="question"
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
            </div>
            <div>
              <StyledLabel htmlFor="answer">Enter the correct answer:</StyledLabel>
              <InputField
                id="answer"
                name="answer"
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
              />
            </div>
            <Button
              type="submit"
              onFocus={() => handleButtonSpeech('Add Question')}
              onMouseEnter={() => handleButtonSpeech('Add Question')}
            >
              Add Question
            </Button>
          </form>

          <FeedbackMessage>{feedback}</FeedbackMessage>

          <Header2>Choose a question:</Header2>
          <ul>
            {questionsList.map((q, index) => (
              <li key={index}>
                <Button
                  onClick={() => setSelectedQuestion(q.question)}
                  onFocus={() => handleButtonSpeech(q.question)}
                  onMouseEnter={() => handleButtonSpeech(q.question)}
                >
                  {q.question}
                </Button>
              </li>
            ))}
          </ul>

          {selectedQuestion && (
            <form onSubmit={handleCheckAnswer}>
              <SubHeader>{selectedQuestion}</SubHeader>
              <StyledLabel htmlFor="userAnswer">Your answer:</StyledLabel>
              <InputField
                id="userAnswer"
                name="userAnswer"
                type="text"
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
              />
              <Button
                type="submit"
                onFocus={() => handleButtonSpeech('Submit Answer')}
                onMouseEnter={() => handleButtonSpeech('Submit Answer')}
              >
                Submit Answer
              </Button>
            </form>
          )}

          {isCorrect !== null && (
            <FeedbackMessage color={isCorrect ? 'green' : 'red'}>
              {feedback}
            </FeedbackMessage>
          )}
        </Card>

        {/* -------------------- Mind Mapping Section -------------------- */}
        <Card>
          <Header2>Mind Mapping</Header2>
          {/* Control Buttons */}
          <ButtonContainer>
            <Button
              onClick={addNodeHandler}
              onFocus={() => handleButtonSpeech('Add Concept')}
              onMouseEnter={() => handleButtonSpeech('Add Concept')}
            >
              <FiPlus style={{ marginRight: '5px' }} /> Add Concept
            </Button>
            <Button
              onClick={removeSelectedElementsHandler}
              bgColor="#e74c3c"
              onFocus={() => handleButtonSpeech('Remove Selected Elements')}
              onMouseEnter={() => handleButtonSpeech('Remove Selected Elements')}
            >
              <FiTrash2 style={{ marginRight: '5px' }} /> Remove
            </Button>
            <Button
              onClick={saveMindMap}
              bgColor="#27ae60"
              onFocus={() => handleButtonSpeech('Save Mind Map')}
              onMouseEnter={() => handleButtonSpeech('Save Mind Map')}
            >
              <FiSave style={{ marginRight: '5px' }} /> Save
            </Button>
            <Button
              onClick={exportAsImage}
              bgColor="#2980b9"
              onFocus={() => handleButtonSpeech('Export as Image')}
              onMouseEnter={() => handleButtonSpeech('Export as Image')}
            >
              <FiDownload style={{ marginRight: '5px' }} /> Export as Image
            </Button>
            <Button
              onClick={exportAsPDF}
              bgColor="#8e44ad"
              onFocus={() => handleButtonSpeech('Export as PDF')}
              onMouseEnter={() => handleButtonSpeech('Export as PDF')}
            >
              <FiDownload style={{ marginRight: '5px' }} /> Export as PDF
            </Button>
          </ButtonContainer>

          {/* React Flow Diagram */}
          <MindMapContainer id="mind-map-container">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onLoad={onLoad}
              nodeTypes={nodeTypes}
              onNodeClick={onNodeClick}
              snapToGrid={true}
              snapGrid={[15, 15]}
              style={{ background: theme === 'light' ? '#f0f0f0' : '#1e1e1e', height: '100%' }}
            >
              <MiniMap
                nodeColor={(node) => {
                  switch (node.type) {
                    case 'input':
                      return 'blue';
                    case 'output':
                      return 'green';
                    default:
                      return '#FFCC00';
                  }
                }}
              />
              <Controls />
              <Background color="#aaa" gap={16} />
            </ReactFlow>

            {/* -------------------- Speech Recognition Controls -------------------- */}
            {selectedNodeId && (
              <SpeechRecognitionComponent
                onTranscript={(transcript) => {
                  console.log('Transcript received:', transcript); // Debugging
                  setNodes((nds) =>
                    nds.map((node) => {
                      if (node.id === selectedNodeId) {
                        console.log(`Updating node ${node.id} with label:`, transcript); // Debugging
                        return {
                          ...node,
                          data: { ...node.data, label: transcript },
                        };
                      }
                      return node;
                    })
                  );
                }}
                handleButtonSpeech={handleButtonSpeech} // Pass the function as a prop
              />
            )}

            {/* Display feedback messages */}
            {isListening && (
              <FeedbackMessage color="green">
                🎤 Listening... Please speak into your microphone.
              </FeedbackMessage>
            )}
            {feedback && <FeedbackMessage>{feedback}</FeedbackMessage>}
            {speechError && <FeedbackMessage color="red">{speechError}</FeedbackMessage>}
          </MindMapContainer>
        </Card>
        <Card>
          
        </Card>
        <Card>
          
        </Card>
        <Footer>
  &copy; {new Date().getFullYear()} LexiSmart. All rights reserved. | Developed by Samay Bhojwani
  <div style={{ marginTop: '10px' }}>
    <a href="https://www.linkedin.com/in/samay-bhojwani-032060260/" aria-label="LinkedIn" style={{ marginLeft: '10px' }}>
      <FaLinkedin />
    </a>
  </div>
</Footer>
      </AppContainer>
      
     

    </ThemeProvider>
  );
}

export default App;
