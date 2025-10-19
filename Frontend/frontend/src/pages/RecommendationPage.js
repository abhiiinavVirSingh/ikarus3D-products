import React, { useState } from 'react';
import axios from 'axios';
import ProductCard from '../components/ProductCard'; // Assuming you create this
import './RecommendationPage.css'; // For styling

// Define backend URL - adjust if needed
const API_URL = 'http://localhost:8000'; // Your FastAPI backend

function RecommendationPage() {
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]); // To store user queries and bot responses

  const handleInputChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setRecommendations([]); // Clear previous recommendations
    const userMessage = { type: 'user', text: query };
    setChatHistory(prev => [...prev, userMessage]); // Add user query to chat

    try {
      const response = await axios.post(`${API_URL}/recommend`, { query: query, top_k: 5 });
      setRecommendations(response.data);
      const botMessage = { type: 'bot', recommendations: response.data };
      setChatHistory(prev => [...prev, botMessage]); // Add bot response to chat
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      const errorMessage = err.response?.data?.detail || "Failed to fetch recommendations.";
      setError(errorMessage);
       const errorBotMessage = { type: 'bot', text: `Sorry, I couldn't get recommendations: ${errorMessage}` };
       setChatHistory(prev => [...prev, errorBotMessage]); // Add error to chat
    } finally {
      setIsLoading(false);
      setQuery(''); // Clear input after submit
    }
  };

  return (
    <div className="recommendation-page">
      <h1>Product Recommendations</h1>
      <p>Describe the furniture you're looking for:</p>

      {/* Chat History Display */}
      <div className="chat-history">
         {chatHistory.map((message, index) => (
             <div key={index} className={`message ${message.type}`}>
                 {message.type === 'user' && <p><strong>You:</strong> {message.text}</p>}
                 {message.type === 'bot' && message.text && <p><strong>Bot:</strong> {message.text}</p>}
                 {message.type === 'bot' && message.recommendations && (
                     <>
                        <p><strong>Bot:</strong> Here are some recommendations:</p>
                        <div className="recommendations-grid">
                            {message.recommendations.map(rec => (
                                <ProductCard key={rec.uniq_id} product={rec} />
                            ))}
                        </div>
                     </>
                 )}
             </div>
         ))}
         {isLoading && <div className="message bot"><p><strong>Bot:</strong> Thinking...</p></div>}
      </div>


      {/* Input Form */}
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="e.g., 'modern wooden coffee table'"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Searching...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <p className="error-message">Error: {error}</p>}

      {/* Recommendation Display Area (Optional, if not using chat history display above) */}
      {/* {!isLoading && !error && recommendations.length > 0 && (
        <div className="recommendations-grid">
          {recommendations.map(rec => (
            <ProductCard key={rec.uniq_id} product={rec} />
          ))}
        </div>
      )} */}
    </div>
  );
}

export default RecommendationPage;