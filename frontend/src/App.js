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
import NewChatModal from './components/NewChatModal';
import ApiKeyModal from './components/ApiKeyModal';
import InfoPage from './components/InfoPage';
import SetupPage from './components/SetupPage';
import { UploadCloudIcon } from './components/Icons';

export const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

const EMPTY_DB_STATUS = { exists: false, file_count: 0 };
const EMPTY_MESSAGES = [];
const EMPTY_UPLOAD_QUEUE = [];
const EMPTY_EXPANDED_SOURCES = {};

function App() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [chatsLoaded, setChatsLoaded] = useState(false);

  // Keyed by chat id, so an in-progress upload or streaming response in one
  // chat keeps updating correctly even if the user switches to another chat
  // before it finishes.
  const [messagesByChat, setMessagesByChat] = useState({});
  const [uploadQueueByChat, setUploadQueueByChat] = useState({});
  const [loadingByChat, setLoadingByChat] = useState({});
  const [dbStatusByChat, setDbStatusByChat] = useState({});
  const [expandedSourcesByChat, setExpandedSourcesByChat] = useState({});

  const [input, setInput] = useState('');
  const [showLibrary, setShowLibrary] = useState(false);
  const [viewingSource, setViewingSource] = useState(null);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyStatus, setApiKeyStatus] = useState(null);
  const [view, setView] = useState('chat'); // 'chat' | 'info' | 'setup'

  const messagesEndRef = useRef(null);
  const uploadIntervalsRef = useRef({});
  const abortControllersRef = useRef({});
  const loadedMessagesRef = useRef(new Set());

  const messages = messagesByChat[activeChatId] || EMPTY_MESSAGES;
  const uploadQueue = uploadQueueByChat[activeChatId] || EMPTY_UPLOAD_QUEUE;
  const loading = loadingByChat[activeChatId] || false;
  const dbStatus = dbStatusByChat[activeChatId] || EMPTY_DB_STATUS;
  const expandedSources = expandedSourcesByChat[activeChatId] || EMPTY_EXPANDED_SOURCES;

  const updateMessages = (chatId, updater) => {
    setMessagesByChat(prev => ({
      ...prev,
      [chatId]: typeof updater === 'function' ? updater(prev[chatId] || []) : updater
    }));
  };

  // Persists the full message list at "settled" points only (a message
  // finishing, not each streamed chunk) - strips the transient `streaming`
  // flag so a reload never resumes into a stuck in-progress bubble.
  const persistMessages = (chatId, messages) => {
    const toSave = messages.map(({ streaming, ...rest }) => rest);
    axios.put(`${API_BASE}/chats/${chatId}/messages`, { messages: toSave })
      .catch(error => console.error('failed to persist messages:', error));
  };

  // Loads a chat's saved history the first time it becomes active. Once
  // loaded, messagesByChat is the source of truth for the session - later
  // switches back to this chat don't re-fetch and clobber local state.
  const loadMessages = async (chatId) => {
    if (loadedMessagesRef.current.has(chatId)) return;
    loadedMessagesRef.current.add(chatId);
    try {
      const response = await axios.get(`${API_BASE}/chats/${chatId}/messages`);
      setMessagesByChat(prev => ({ ...prev, [chatId]: response.data.messages || [] }));
    } catch (error) {
      console.error('failed to load messages:', error);
      loadedMessagesRef.current.delete(chatId);
    }
  };

  const updateUploadQueue = (chatId, updater) => {
    setUploadQueueByChat(prev => ({
      ...prev,
      [chatId]: typeof updater === 'function' ? updater(prev[chatId] || []) : updater
    }));
  };

  const setChatLoading = (chatId, value) => {
    setLoadingByChat(prev => ({ ...prev, [chatId]: value }));
  };

  const refreshChats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/chats`);
      setChats(response.data);
      return response.data;
    } catch (error) {
      console.error('failed to load chats:', error);
      return null;
    }
  };

  const refreshApiKeyStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/settings/api-key`);
      setApiKeyStatus(response.data);
    } catch (error) {
      console.error('failed to load api key status:', error);
    }
  };

  // Load the chat list on mount, and open the first chat if any exist
  useEffect(() => {
    (async () => {
      const fetched = await refreshChats();
      if (fetched && fetched.length > 0) {
        setActiveChatId(fetched[0].id);
        checkStatuses(fetched[0].id);
        loadMessages(fetched[0].id);
      }
      setChatsLoaded(true);
    })();
    refreshApiKeyStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Scroll to bottom when the active chat's messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup upload polling and in-flight chat requests on unmount
  useEffect(() => {
    return () => {
      Object.values(uploadIntervalsRef.current).forEach(interval => {
        clearInterval(interval);
      });
      Object.values(abortControllersRef.current).forEach(controller => {
        controller.abort();
      });
    };
  }, []);

  const checkStatuses = async (chatId) => {
    try {
      const dbResponse = await axios.get(`${API_BASE}/chats/${chatId}/database/status`);
      setDbStatusByChat(prev => ({ ...prev, [chatId]: dbResponse.data }));
    } catch (error) {
      console.error('failed to check statuses:', error);
    }
  };

  const switchChat = (chatId) => {
    setActiveChatId(chatId);
    setShowLibrary(false);
    setViewingSource(null);
    setView('chat');
    checkStatuses(chatId);
    loadMessages(chatId);
  };

  const handleCreateChat = async (name, copyFrom) => {
    const response = await axios.post(`${API_BASE}/chats`, { name, copy_from: copyFrom || null });
    const newChat = response.data;
    setChats(prev => [...prev, newChat]);
    switchChat(newChat.id);
    setShowNewChatModal(false);
  };

  const handleDeleteChat = async (chatId) => {
    const remaining = chats.filter(c => c.id !== chatId);
    try {
      await axios.delete(`${API_BASE}/chats/${chatId}`);
    } catch (error) {
      console.error('failed to delete chat:', error);
      return;
    }

    setChats(remaining);
    [setMessagesByChat, setUploadQueueByChat, setLoadingByChat, setDbStatusByChat, setExpandedSourcesByChat].forEach(setter => {
      setter(prev => {
        const next = { ...prev };
        delete next[chatId];
        return next;
      });
    });

    if (activeChatId === chatId) {
      if (remaining.length > 0) {
        switchChat(remaining[0].id);
      } else {
        setActiveChatId(null);
      }
    }
  };

  const pollUploadStatus = async (chatId, uploadId) => {
    try {
      const response = await axios.get(`${API_BASE}/documents/upload/status/${uploadId}`);
      const status = response.data;

      updateUploadQueue(chatId, prev => prev.map(upload =>
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

      if (status.status === 'completed') {
        if (uploadIntervalsRef.current[uploadId]) {
          clearInterval(uploadIntervalsRef.current[uploadId]);
          delete uploadIntervalsRef.current[uploadId];
        }
        updateMessages(chatId, prev => {
          const next = [...prev, {
            role: 'system',
            content: `Successfully processed ${status.total_files} document${status.total_files === 1 ? '' : 's'}`
          }];
          persistMessages(chatId, next);
          return next;
        });
        checkStatuses(chatId);
        refreshChats();

        setTimeout(() => {
          updateUploadQueue(chatId, prev => prev.filter(u => u.uploadId !== uploadId));
        }, 3000);
      } else if (status.status === 'error') {
        if (uploadIntervalsRef.current[uploadId]) {
          clearInterval(uploadIntervalsRef.current[uploadId]);
          delete uploadIntervalsRef.current[uploadId];
        }
        updateMessages(chatId, prev => {
          const next = [...prev, {
            role: 'system',
            content: `Upload failed: ${status.message}`
          }];
          persistMessages(chatId, next);
          return next;
        });

        setTimeout(() => {
          updateUploadQueue(chatId, prev => prev.filter(u => u.uploadId !== uploadId));
        }, 5000);
      }
    } catch (error) {
      console.error('failed to poll upload status:', error);
      if (uploadIntervalsRef.current[uploadId]) {
        clearInterval(uploadIntervalsRef.current[uploadId]);
        delete uploadIntervalsRef.current[uploadId];
      }
      setTimeout(() => {
        updateUploadQueue(chatId, prev => prev.filter(u => u.uploadId !== uploadId));
      }, 3000);
    }
  };

  const handleFileUpload = async (files) => {
    const chatId = activeChatId;
    if (!chatId || !files || files.length === 0) return;

    const tempId = `temp-${Date.now()}`;
    const fileNames = Array.from(files).map(f => f.name).join(', ');

    updateUploadQueue(chatId, prev => [...prev, {
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

      const response = await axios.post(`${API_BASE}/chats/${chatId}/documents/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          updateUploadQueue(chatId, prev => prev.map(upload =>
            upload.uploadId === tempId
              ? { ...upload, progress: Math.min(percentCompleted, 10), message: 'Uploading files...' }
              : upload
          ));
        }
      });

      const uploadId = response.data.upload_id;

      updateUploadQueue(chatId, prev => prev.map(upload =>
        upload.uploadId === tempId ? { ...upload, uploadId: uploadId } : upload
      ));

      uploadIntervalsRef.current[uploadId] = setInterval(() => {
        pollUploadStatus(chatId, uploadId);
      }, 500);

      pollUploadStatus(chatId, uploadId);

    } catch (error) {
      updateUploadQueue(chatId, prev => prev.filter(u => u.uploadId !== tempId));
      updateMessages(chatId, prev => {
        const next = [...prev, {
          role: 'system',
          content: `Upload failed: ${error.response?.data?.detail || error.message}`
        }];
        persistMessages(chatId, next);
        return next;
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const chatId = activeChatId;
    if (!chatId || !input.trim() || loadingByChat[chatId]) return;

    const priorMessages = messagesByChat[chatId] || [];
    const currentInput = input;

    const userMessage = { role: 'user', content: currentInput };
    updateMessages(chatId, prev => {
      const next = [...prev, userMessage];
      persistMessages(chatId, next);
      return next;
    });
    setChatLoading(chatId, true);

    const assistantMessage = {
      role: 'assistant',
      content: '',
      sources: null,
      response_time: null,
      streaming: true
    };
    updateMessages(chatId, prev => [...prev, assistantMessage]);

    const controller = new AbortController();
    abortControllersRef.current[chatId] = controller;

    try {
      const conversationHistory = priorMessages
        .filter(msg => msg.role !== 'system')
        .slice(-6) // Last 3 Q&A pairs
        .reduce((acc, msg, index, arr) => {
          if (msg.role === 'user' && arr[index + 1]?.role === 'assistant') {
            acc.push([msg.content, arr[index + 1].content]);
          }
          return acc;
        }, []);

      const response = await fetch(`${API_BASE}/chats/${chatId}/qa/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: currentInput,
          conversation_history: conversationHistory
        }),
        signal: controller.signal
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

                updateMessages(chatId, prev => {
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
                updateMessages(chatId, prev => {
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
                updateMessages(chatId, prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage.role === 'assistant' && lastMessage.streaming) {
                    newMessages[newMessages.length - 1] = {
                      ...lastMessage,
                      streaming: false,
                      response_time: data.response_time,
                      content: accumulatedContent,
                      sources: lastMessage.sources
                    };
                  }
                  persistMessages(chatId, newMessages);
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
      if (error.name === 'AbortError') {
        // Cancelled by the user - keep whatever content already streamed in,
        // just stop treating it as in-progress.
        updateMessages(chatId, prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage?.role === 'assistant' && lastMessage.streaming) {
            newMessages[newMessages.length - 1] = { ...lastMessage, streaming: false };
          }
          persistMessages(chatId, newMessages);
          return newMessages;
        });
      } else {
        updateMessages(chatId, prev => {
          const newMessages = prev.slice(0, -1);
          const next = [...newMessages, {
            role: 'assistant',
            content: `Error: ${error.message}`
          }];
          persistMessages(chatId, next);
          return next;
        });
      }
    }

    delete abortControllersRef.current[chatId];
    setChatLoading(chatId, false);
    if (chatId === activeChatId) setInput('');
  };

  const handleCancelResponse = () => {
    if (!activeChatId) return;
    abortControllersRef.current[activeChatId]?.abort();
  };

  const clearChat = () => {
    if (!activeChatId) return;
    updateMessages(activeChatId, []);
    persistMessages(activeChatId, []);
  };

  const resetDatabase = async () => {
    const chatId = activeChatId;
    if (!chatId) return;

    (uploadQueueByChat[chatId] || []).forEach(upload => {
      if (uploadIntervalsRef.current[upload.uploadId]) {
        clearInterval(uploadIntervalsRef.current[upload.uploadId]);
        delete uploadIntervalsRef.current[upload.uploadId];
      }
    });
    updateUploadQueue(chatId, []);

    setChatLoading(chatId, true);
    try {
      await axios.post(`${API_BASE}/chats/${chatId}/database/reset`, {}, {
        timeout: 10000
      });
      const resetMessage = [{ role: 'system', content: 'Database reset successfully' }];
      updateMessages(chatId, resetMessage);
      persistMessages(chatId, resetMessage);
      checkStatuses(chatId);
      refreshChats();
    } catch (error) {
      const errorMsg = error.code === 'ECONNABORTED'
        ? 'Reset timed out - please try again'
        : `Reset failed: ${error.response?.data?.detail || error.message}`;
      updateMessages(chatId, prev => {
        const next = [...prev, { role: 'system', content: errorMsg }];
        persistMessages(chatId, next);
        return next;
      });
    }
    setChatLoading(chatId, false);
  };

  const toggleSources = (index) => {
    if (!activeChatId) return;
    setExpandedSourcesByChat(prev => ({
      ...prev,
      [activeChatId]: { ...(prev[activeChatId] || {}), [index]: !(prev[activeChatId] || {})[index] }
    }));
  };

  const canChat = !!activeChatId && dbStatus.exists;

  const renderChatView = () => (
    <>
      <div className="title">document q&amp;a</div>
      <div className="subtitle">ask questions grounded in your uploaded pdfs</div>

      {!chatsLoaded ? null : !activeChatId ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <UploadCloudIcon size={22} />
          </div>
          <h2>create your first chat</h2>
          <p>each chat has its own set of documents. start one empty, or copy an existing chat's documents.</p>
          <button className="nav-item primary-action" onClick={() => setShowNewChatModal(true)}>
            new chat
          </button>
        </div>
      ) : !canChat && messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <UploadCloudIcon size={22} />
          </div>
          <h2>upload documents to begin</h2>
          <p>add one or more pdfs and athena will index them for grounded, cited answers.</p>
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
            onCancel={handleCancelResponse}
            canChat={canChat}
            loading={loading}
          />
          {loading && !messages[messages.length - 1]?.streaming && (
            <div className="loading">working...</div>
          )}
        </>
      )}
    </>
  );

  return (
    <div className="app">
      <div className="container">
        <Sidebar
          chats={chats}
          activeChatId={activeChatId}
          onSwitchChat={switchChat}
          onNewChat={() => setShowNewChatModal(true)}
          onDeleteChat={handleDeleteChat}
          onClearChat={clearChat}
          onResetDatabase={resetDatabase}
          onOpenLibrary={() => setShowLibrary(true)}
          onOpenApiKey={() => setShowApiKeyModal(true)}
          apiKeyStatus={apiKeyStatus}
          view={view}
          onOpenInfo={() => setView('info')}
          onOpenSetup={() => setView('setup')}
          loading={loading}
          dbStatus={dbStatus}
        />

        {showNewChatModal && (
          <NewChatModal
            chats={chats}
            onCreate={handleCreateChat}
            onClose={() => setShowNewChatModal(false)}
          />
        )}

        {showApiKeyModal && (
          <ApiKeyModal
            onChanged={refreshApiKeyStatus}
            onClose={() => setShowApiKeyModal(false)}
          />
        )}

        {showLibrary && activeChatId && (
          <DocumentLibraryModal
            chatId={activeChatId}
            onClose={() => setShowLibrary(false)}
            onDocumentDeleted={() => { checkStatuses(activeChatId); refreshChats(); }}
          />
        )}

        {viewingSource && (
          <DocumentViewerModal
            chatId={activeChatId}
            name={viewingSource.name}
            startIndex={viewingSource.start_index}
            length={viewingSource.length}
            onClose={() => setViewingSource(null)}
          />
        )}

        <div className="main">
          {view === 'info' ? (
            <InfoPage />
          ) : view === 'setup' ? (
            <SetupPage
              apiKeyStatus={apiKeyStatus}
              onOpenApiKey={() => setShowApiKeyModal(true)}
              onNewChat={() => setShowNewChatModal(true)}
            />
          ) : (
            renderChatView()
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
