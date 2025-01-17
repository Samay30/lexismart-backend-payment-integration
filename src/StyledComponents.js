// StyledComponents.js
// StyledComponents.js
import styled from 'styled-components';

// Centered H1 Component
export const Header = styled.h1`
  text-align: center;
  color: ${({ theme }) => theme.textColor};
`;

export const Header2 = styled.h2`
  text-align: center;
  color: ${({ theme }) => theme.textColor};
`;

export const Header3 = styled.h3`
  text-align: center;
  color: ${({ theme }) => theme.textColor};
`;



// Container for the entire app
export const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  padding: 20px;
  box-sizing: border-box;
  background-color: ${({ theme }) => theme.background};
  min-height: 100vh;
  transition: background-color 0.3s ease, color 0.3s ease;

  @media (min-width: 768px) {
    padding: 40px;
  }
`;

// Card component for different sections
export const Card = styled.div`
  background-color: ${({ theme }) => theme.cardBackground};
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
  color: ${({ theme }) => theme.textColor};
  transition: background-color 0.3s ease, color 0.3s ease;

  @media (min-width: 768px) {
    padding: 30px;
  }
`;

// Flex container for buttons
export const ButtonContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 10px;

  @media (min-width: 600px) {
    flex-direction: row;
  }
`;

// Button styling
export const Button = styled.button`
  flex: 1;
  padding: 12px;
  background-color: ${({ bgColor, theme }) =>
    bgColor || theme.buttonBackground};
  color: ${({ theme }) => theme.buttonColor};
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s ease, box-shadow 0.3s ease;

  &:disabled {
    background-color: ${({ theme }) => theme.buttonDisabledBackground};
    cursor: not-allowed;
  }

  /* Active state for touch feedback */
  &:active {
    transform: scale(0.98);
    box-shadow: 0px 2px 4px ${({ theme }) => theme.buttonActiveShadow};
  }

  @media (max-width: 599px) {
    padding: 16px;
    font-size: 1.1rem;
  }
`;

// Textarea styling
export const TextArea = styled.textarea`
  width: 100%;
  min-height: 100px;
  padding: 10px;
  font-size: 1rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  resize: vertical;
  box-sizing: border-box;
  color: ${({ theme }) => theme.textColor};
  background-color: ${({ theme }) => theme.cardBackground};
  transition: background-color 0.3s ease, color 0.3s ease;

  @media (max-width: 599px) {
    min-height: 80px;
    font-size: 0.9rem;
  }
`;

// Input field styling
export const InputField = styled.input`
  width: 100%;
  padding: 10px;
  font-size: 1rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  box-sizing: border-box;
  color: ${({ theme }) => theme.textColor};
  background-color: ${({ theme }) => theme.cardBackground};
  transition: background-color 0.3s ease, color 0.3s ease;

  @media (max-width: 599px) {
    font-size: 0.9rem;
  }
`;

// Select field styling
export const SelectField = styled.select`
  width: 100%;
  padding: 10px;
  font-size: 1rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  box-sizing: border-box;
  color: ${({ theme }) => theme.textColor};
  background-color: ${({ theme }) => theme.cardBackground};
  transition: background-color 0.3s ease, color 0.3s ease;

  @media (max-width: 599px) {
    font-size: 0.9rem;
  }
`;

// Mind Map Container
export const MindMapContainer = styled.div`
  height: 60vh;
  width: 100%;
  position: relative;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-sizing: border-box;

  @media (max-width: 599px) {
    height: 50vh;
  }
`;

// Highlighted Text Styling (now as h2)
export const SummaryHeader = styled.h2`
  margin-top: 10px;
  font-size: 1.5rem;
  line-height: 3;
  color: ${({ theme }) => theme.textColor};
  transition: color 0.3s ease;

  .highlight {
    background-color: ${({ theme }) => theme.highlightColor};
  }

  @media (max-width: 599px) {
    font-size: 1.2rem;
  }
`;

// Sub Header (styled h3)
export const SubHeader = styled.h3`
  margin-top: 10px;
  font-size: 1.25rem;
  color: ${({ theme }) => theme.textColor};
  transition: color 0.3s ease;

  @media (max-width: 599px) {
    font-size: 1rem;
  }
`;

// Styled Label
export const StyledLabel = styled.label`
  display: block;
  margin-top: 10px;
  font-size: 1rem;
  color: ${({ theme }) => theme.textColor};
  transition: color 0.3s ease;

  @media (max-width: 599px) {
    font-size: 0.9rem;
  }
`;

// Feedback Message Styling
export const FeedbackMessage = styled.p`
  margin-top: 10px;
  font-size: 1rem;
  color: ${({ color, theme }) => color || theme.textColor};
  transition: color 0.3s ease;

  @media (max-width: 599px) {
    font-size: 0.9rem;
  }
`;

// ... existing styled-components

// Footer Component
export const Footer = styled.footer`
  background-color: ${({ theme }) => theme.footerBackground};
  color: ${({ theme }) => theme.footerText};
  padding: 20px;
  text-align: center;
  width: 100%;
  box-sizing: border-box;
  margin-top: auto; /* Pushes the footer to the bottom */

  a {
    color: ${({ theme }) => theme.footerLink};
    margin: 0 10px;
    text-decoration: none;
    font-weight: bold;

    &:hover {
      text-decoration: underline;
    }
  }

  @media (max-width: 600px) {
    padding: 15px;
    font-size: 0.9rem;
  }
`;

