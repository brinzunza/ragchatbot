import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dbStatus, setDbStatus] = useState({ exists: false, file_count: 0 });
  const [uploadProgress, setUploadProgress] = useState(null);
  const messagesEndRef = useRef(null);
  const uploadIntervalRef = useRef(null);

  // Check database and data status on mount
  useEffect(() => {
    checkStatuses();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup upload polling on unmount
  useEffect(() => {
    return () => {
      if (uploadIntervalRef.current) {
        clearInterval(uploadIntervalRef.current);
      }
    };
  }, []);

  const checkStatuses = async () => {
    try {
      const dbResponse = await axios.get(`${API_BASE}/database/status`);
      setDbStatus(dbResponse.data);
    } catch (error) {
      console.error('failed to check statuses:', error);
    }
  };

  const pollUploadStatus = async (uploadId) => {
    try {
      const response = await axios.get(`${API_BASE}/documents/upload/status/${uploadId}`);
      const status = response.data;

      setUploadProgress({
        progress: status.progress,
        status: status.status,
        message: status.message,
        currentFile: status.current_file,
        totalFiles: status.total_files,
        processedFiles: status.processed_files
      });

      // If completed or error, stop polling
      if (status.status === 'completed') {
        clearInterval(uploadIntervalRef.current);
        uploadIntervalRef.current = null;
        setMessages(prev => [...prev, {
          role: 'system',
          content: `successfully processed ${status.total_files} documents`
        }]);
        checkStatuses();
        setTimeout(() => setUploadProgress(null), 2000);
      } else if (status.status === 'error') {
        clearInterval(uploadIntervalRef.current);
        uploadIntervalRef.current = null;
        setMessages(prev => [...prev, {
          role: 'system',
          content: `upload failed: ${status.message}`
        }]);
        setTimeout(() => setUploadProgress(null), 3000);
      }
    } catch (error) {
      console.error('failed to poll upload status:', error);
      clearInterval(uploadIntervalRef.current);
      uploadIntervalRef.current = null;
      setUploadProgress(null);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // Clear any previous upload progress
    if (uploadIntervalRef.current) {
      clearInterval(uploadIntervalRef.current);
    }

    // Initialize upload progress
    setUploadProgress({
      progress: 0,
      status: 'uploading',
      message: 'uploading files...',
      currentFile: null,
      totalFiles: files.length,
      processedFiles: 0
    });

    try {
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('files', file);
      });

      // Upload with progress tracking
      const response = await axios.post(`${API_BASE}/documents/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(prev => ({
            ...prev,
            progress: Math.min(percentCompleted, 10), // Cap at 10% for upload phase
            message: 'uploading files...'
          }));
        }
      });

      // Start polling for background processing status
      const uploadId = response.data.upload_id;
      uploadIntervalRef.current = setInterval(() => {
        pollUploadStatus(uploadId);
      }, 500); // Poll every 500ms

      // Immediately poll once
      pollUploadStatus(uploadId);

    } catch (error) {
      setUploadProgress(null);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `upload failed: ${error.response?.data?.detail || error.message}`
      }]);
    }

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
                  disabled={uploadProgress !== null}
                />
              </div>
            </div>
          )}

          {/* Upload progress bar */}
          {uploadProgress && (
            <div className="upload-progress">
              <div className="progress-info">
                <span className="progress-message">{uploadProgress.message}</span>
                <span className="progress-percentage">{uploadProgress.progress}%</span>
              </div>
              <div className="progress-bar-container">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${uploadProgress.progress}%` }}
                />
              </div>
              {uploadProgress.currentFile && (
                <div className="progress-file">
                  processing: {uploadProgress.currentFile}
                </div>
              )}
              {uploadProgress.totalFiles > 0 && (
                <div className="progress-count">
                  {uploadProgress.processedFiles} / {uploadProgress.totalFiles} files
                </div>
              )}
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