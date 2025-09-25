import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dbStatus, setDbStatus] = useState({ exists: false, file_count: 0 });
  const messagesEndRef = useRef(null);

  // Check database and data status on mount
  useEffect(() => {
    checkStatuses();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkStatuses = async () => {
    try {
      const dbResponse = await axios.get(`${API_BASE}/database/status`);
      setDbStatus(dbResponse.data);
    } catch (error) {
      console.error('failed to check statuses:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setLoading(true);
    try {
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('files', file);
      });
      await axios.post(`${API_BASE}/documents/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMessages([{ role: 'system', content: 'documents uploaded and database created' }]);
      checkStatuses();
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'system',
        content: `upload failed: ${error.response?.data?.detail || error.message}`
      }]);
    }
    setLoading(false);
    event.target.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    // Add a placeholder assistant message for streaming
    const assistantMessage = {
      role: 'assistant',
      content: '',
      response_time: null,
      streaming: true
    };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      const conversationHistory = messages
        .filter(msg => msg.role !== 'system')
        .slice(-6) // Last 3 Q&A pairs
        .reduce((acc, msg, index, arr) => {
          if (msg.role === 'user' && arr[index + 1]?.role === 'assistant') {
            acc.push([msg.content, arr[index + 1].content]);
          }
          return acc;
        }, []);

      // Use fetch for Server-Sent Events streaming
      const response = await fetch(`${API_BASE}/qa/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: input,
          conversation_history: conversationHistory
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ') && line.trim().length > 6) {
            try {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;

              const data = JSON.parse(jsonStr);

              if (data.type === 'chunk' && data.content) {
                accumulatedContent += data.content;

                // Update the streaming message with accumulated content
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage.role === 'assistant' && lastMessage.streaming) {
                    newMessages[newMessages.length - 1] = {
                      ...lastMessage,
                      content: accumulatedContent
                    };
                  }
                  return newMessages;
                });
              } else if (data.type === 'complete') {
                // Mark streaming as complete and add response time
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage.role === 'assistant' && lastMessage.streaming) {
                    newMessages[newMessages.length - 1] = {
                      ...lastMessage,
                      streaming: false,
                      response_time: data.response_time,
                      content: accumulatedContent
                    };
                  }
                  return newMessages;
                });
                break;
              } else if (data.type === 'error') {
                throw new Error(data.message);
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', line, parseError);
            }
          }
        }
      }
    } catch (error) {
      // Remove the streaming message and add error message
      setMessages(prev => {
        const newMessages = prev.slice(0, -1);
        return [...newMessages, {
          role: 'assistant',
          content: `error: ${error.message}`
        }];
      });
    }

    setLoading(false);
    setInput('');
  };

  const clearChat = () => {
    setMessages([]);
  };

  const resetDatabase = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/database/reset`);
      setMessages([{ role: 'system', content: 'database reset successfully' }]);
      checkStatuses();
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'system',
        content: `reset failed: ${error.response?.data?.detail || error.message}`
      }]);
    }
    setLoading(false);
  };

  const canChat = dbStatus.exists;

  return (
    <div className="app">
      <div className="container">
        {/* Sidebar */}
        <div className="sidebar">
          <div className="section">
            <p>ai assistant</p>
          </div>

          <div className="section">
            <button onClick={clearChat} className="text-button">
              clear chat
            </button>
            <button onClick={resetDatabase} className="text-button">
              reset database
            </button>
          </div>

          <div className="section">
            <p>pdf document analysis</p>
          </div>
        </div>

        {/* Main content */}
        <div className="main">
          <div className="title">
            document q&a
          </div>

          {/* Status and file upload */}
          {!canChat && (
            <div className="upload-section">
              <div>
                <p>upload pdf documents to begin</p>
                <input
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={handleFileUpload}
                  className="file-input"
                />
              </div>
            </div>
          )}

          {/* Chat messages */}
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
                  {message.content}
                  {message.response_time && (
                    <div className="response-time">
                      {message.response_time.toFixed(2)}s
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat input */}
          <form onSubmit={handleSubmit} className="chat-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={canChat ? 'ask about your documents' : 'upload pdf files first'}
              disabled={!canChat || loading}
              className="chat-input"
            />
          </form>

          {loading && (
            <div className="loading">
              processing...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;