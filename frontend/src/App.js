import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import Sidebar from './components/Sidebar';
import UploadDropzone from './components/UploadDropzone';
import UploadQueue from './components/UploadQueue';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';
import DocumentLibraryModal from './components/DocumentLibraryModal';
import DocumentViewerModal from './components/DocumentViewerModal';
import { UploadCloudIcon } from './components/Icons';

export const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dbStatus, setDbStatus] = useState({ exists: false, file_count: 0 });
  const [uploadQueue, setUploadQueue] = useState([]);
  const [expandedSources, setExpandedSources] = useState({});
  const [showLibrary, setShowLibrary] = useState(false);
  const [viewingSource, setViewingSource] = useState(null);
  const messagesEndRef = useRef(null);
  const uploadIntervalsRef = useRef({});

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
      Object.values(uploadIntervalsRef.current).forEach(interval => {
        clearInterval(interval);
      });
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

      // Update the specific upload in the queue
      setUploadQueue(prev => prev.map(upload =>
        upload.uploadId === uploadId
          ? {
              ...upload,
              progress: status.progress,
              status: status.status,
              message: status.message,
              currentFile: status.current_file,
              totalFiles: status.total_files,
              processedFiles: status.processed_files
            }
          : upload
      ));

      // If completed or error, stop polling and remove from queue after delay
      if (status.status === 'completed') {
        if (uploadIntervalsRef.current[uploadId]) {
          clearInterval(uploadIntervalsRef.current[uploadId]);
          delete uploadIntervalsRef.current[uploadId];
        }
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Successfully processed ${status.total_files} document${status.total_files === 1 ? '' : 's'}`
        }]);
        checkStatuses();

        // Remove from queue after 3 seconds
        setTimeout(() => {
          setUploadQueue(prev => prev.filter(u => u.uploadId !== uploadId));
        }, 3000);
      } else if (status.status === 'error') {
        if (uploadIntervalsRef.current[uploadId]) {
          clearInterval(uploadIntervalsRef.current[uploadId]);
          delete uploadIntervalsRef.current[uploadId];
        }
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Upload failed: ${status.message}`
        }]);

        // Remove from queue after 5 seconds
        setTimeout(() => {
          setUploadQueue(prev => prev.filter(u => u.uploadId !== uploadId));
        }, 5000);
      }
    } catch (error) {
      console.error('failed to poll upload status:', error);
      if (uploadIntervalsRef.current[uploadId]) {
        clearInterval(uploadIntervalsRef.current[uploadId]);
        delete uploadIntervalsRef.current[uploadId];
      }
      // Remove failed upload from queue
      setTimeout(() => {
        setUploadQueue(prev => prev.filter(u => u.uploadId !== uploadId));
      }, 3000);
    }
  };

  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;

    // Generate a temporary ID for this upload
    const tempId = `temp-${Date.now()}`;
    const fileNames = Array.from(files).map(f => f.name).join(', ');

    // Add upload to queue immediately
    setUploadQueue(prev => [...prev, {
      uploadId: tempId,
      progress: 0,
      status: 'uploading',
      message: 'Uploading files...',
      currentFile: null,
      totalFiles: files.length,
      processedFiles: 0,
      fileNames: fileNames
    }]);

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
          setUploadQueue(prev => prev.map(upload =>
            upload.uploadId === tempId
              ? {
                  ...upload,
                  progress: Math.min(percentCompleted, 10),
                  message: 'Uploading files...'
                }
              : upload
          ));
        }
      });

      // Get the real upload ID from backend
      const uploadId = response.data.upload_id;

      // Update the queue with the real upload ID
      setUploadQueue(prev => prev.map(upload =>
        upload.uploadId === tempId
          ? { ...upload, uploadId: uploadId }
          : upload
      ));

      // Start polling for background processing status
      uploadIntervalsRef.current[uploadId] = setInterval(() => {
        pollUploadStatus(uploadId);
      }, 500); // Poll every 500ms

      // Immediately poll once
      pollUploadStatus(uploadId);

    } catch (error) {
      // Remove from queue on error
      setUploadQueue(prev => prev.filter(u => u.uploadId !== tempId));
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Upload failed: ${error.response?.data?.detail || error.message}`
      }]);
    }
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
      sources: null,
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
              } else if (data.type === 'sources' && data.content) {
                // Store sources separately without adding to content
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage.role === 'assistant' && lastMessage.streaming) {
                    newMessages[newMessages.length - 1] = {
                      ...lastMessage,
                      sources: data.content
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
                      content: accumulatedContent,
                      sources: lastMessage.sources // Explicitly preserve sources
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
          content: `Error: ${error.message}`
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
    // Stop all upload polling
    Object.values(uploadIntervalsRef.current).forEach(interval => {
      clearInterval(interval);
    });
    uploadIntervalsRef.current = {};

    // Clear upload queue
    setUploadQueue([]);

    setLoading(true);
    try {
      await axios.post(`${API_BASE}/database/reset`, {}, {
        timeout: 10000 // 10 second timeout
      });
      setMessages([{ role: 'system', content: 'Database reset successfully' }]);
      checkStatuses();
    } catch (error) {
      const errorMsg = error.code === 'ECONNABORTED'
        ? 'Reset timed out - please try again'
        : `Reset failed: ${error.response?.data?.detail || error.message}`;
      setMessages(prev => [...prev, {
        role: 'system',
        content: errorMsg
      }]);
    }
    setLoading(false);
  };

  const toggleSources = (index) => {
    setExpandedSources(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const canChat = dbStatus.exists;

  return (
    <div className="app">
      <div className="container">
        <Sidebar
          onClearChat={clearChat}
          onResetDatabase={resetDatabase}
          onOpenLibrary={() => setShowLibrary(true)}
          loading={loading}
          dbStatus={dbStatus}
        />

        {showLibrary && (
          <DocumentLibraryModal
            onClose={() => setShowLibrary(false)}
            onDocumentDeleted={checkStatuses}
          />
        )}

        {viewingSource && (
          <DocumentViewerModal
            name={viewingSource.name}
            startIndex={viewingSource.start_index}
            length={viewingSource.length}
            onClose={() => setViewingSource(null)}
          />
        )}

        <div className="main">
          <div className="title">Document Q&amp;A</div>
          <div className="subtitle">Ask questions grounded in your uploaded PDFs</div>

          {!canChat && messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <UploadCloudIcon size={22} />
              </div>
              <h2>Upload documents to begin</h2>
              <p>Add one or more PDFs and Athena will index them for grounded, cited answers.</p>
              <UploadDropzone onFiles={handleFileUpload} disabled={loading} />
              <UploadQueue uploadQueue={uploadQueue} />
            </div>
          ) : (
            <>
              <UploadDropzone onFiles={handleFileUpload} disabled={loading} compact={canChat} />
              <UploadQueue uploadQueue={uploadQueue} />
              <MessageList
                messages={messages}
                expandedSources={expandedSources}
                onToggleSources={toggleSources}
                onViewSource={setViewingSource}
                messagesEndRef={messagesEndRef}
              />
              <ChatInput
                input={input}
                setInput={setInput}
                onSubmit={handleSubmit}
                canChat={canChat}
                loading={loading}
              />
              {loading && !messages[messages.length - 1]?.streaming && (
                <div className="loading">Working...</div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
