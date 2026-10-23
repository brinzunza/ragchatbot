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
import ContextModal from './components/ContextModal';
import AgentModal from './components/AgentModal';
import InfoPage from './components/InfoPage';
import SetupPage from './components/SetupPage';
import { UploadCloudIcon } from './components/Icons';
import { CONTEXT_USAGE, RESEARCH_USAGE, parseContextCommand, parseResearchCommand } from './commands';
import { canRedoLast, rewindLastExchange } from './chatHistory';

export const API_BASE = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

const EMPTY_DB_STATUS = { exists: false, file_count: 0 };
const EMPTY_MESSAGES = [];
const EMPTY_UPLOAD_QUEUE = [];
const EMPTY_EXPANDED_SOURCES = {};

// Turns whatever a failed request threw into a message the user can act on,
// plus the backend's error code when it sent one (the UI uses it to offer a
// shortcut to the settings).
function readApiError(error) {
  const lostConnection =
    (error instanceof TypeError && /fetch|network|load failed/i.test(error.message)) ||
    error.message === 'Network Error';
  if (lostConnection) {
    return { message: "couldn't reach the server. make sure the backend is running.", code: 'unreachable' };
  }
  const detail = error.response?.data?.detail;
  if (detail && typeof detail === 'object') return { message: detail.message, code: detail.code };
  return { message: (typeof detail === 'string' && detail) || error.message, code: error.errorCode };
}

// Steps still marked in-progress when a message is saved never finished
// (cancelled, or the page was left); store them as stopped so a reload
// doesn't show them working forever.
function settleSteps(message) {
  if (!message.steps) return message;
  return {
    ...message,
    steps: message.steps.map((step) => ({
      ...step,
      status: step.status === 'running' ? 'stopped' : step.status,
      permission: step.permission?.state === 'pending' ? { ...step.permission, state: 'expired' } : step.permission,
    })),
  };
}

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
  const [showContext, setShowContext] = useState(false);
  const [showAgent, setShowAgent] = useState(false);
  const [chatConfig, setChatConfig] = useState(null);
  const [contextUsageByChat, setContextUsageByChat] = useState({});
  const [apiKeyStatus, setApiKeyStatus] = useState(null);
  const [view, setView] = useState('chat'); // 'chat' | 'info' | 'setup'

  const uploadIntervalsRef = useRef({});
  const abortControllersRef = useRef({});
  const loadedMessagesRef = useRef(new Set());

  const messages = messagesByChat[activeChatId] || EMPTY_MESSAGES;
  const uploadQueue = uploadQueueByChat[activeChatId] || EMPTY_UPLOAD_QUEUE;
  const loading = loadingByChat[activeChatId] || false;
  const dbStatus = dbStatusByChat[activeChatId] || EMPTY_DB_STATUS;
  const expandedSources = expandedSourcesByChat[activeChatId] || EMPTY_EXPANDED_SOURCES;

  // The active chat's config decides whether chatting needs documents (the
  // agent can answer without them) and what the input says.
  useEffect(() => {
    setChatConfig(null);
    if (!activeChatId) return undefined;
    let stale = false;
    axios.get(`${API_BASE}/chats/${activeChatId}/config`)
      .then(response => { if (!stale) setChatConfig(response.data); })
      .catch(error => console.error('failed to load chat config:', error));
    return () => { stale = true; };
  }, [activeChatId]);

  // How full the chat's context window is, for the sidebar badge
  const refreshContextUsage = async (chatId) => {
    if (!chatId) return;
    try {
      const response = await axios.get(`${API_BASE}/chats/${chatId}/context/usage`);
      setContextUsageByChat(prev => ({ ...prev, [chatId]: response.data }));
    } catch (error) {
      console.error('failed to load context usage:', error);
    }
  };

  useEffect(() => {
    refreshContextUsage(activeChatId);
  }, [activeChatId]); // eslint-disable-line react-hooks/exhaustive-deps

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
    const toSave = messages.map(({ streaming, ...rest }) => settleSteps(rest));
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
    setShowContext(false);
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
            role: 'error',
            content: `upload failed: ${status.message}`
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
      const { message: failure } = readApiError(error);
      updateMessages(chatId, prev => {
        const next = [...prev, {
          role: 'error',
          content: `upload failed: ${failure}`
        }];
        persistMessages(chatId, next);
        return next;
      });
    }
  };

  // Runs a /context slash command. These never reach the model or the
  // context itself: the command is echoed as a user message, and the
  // outcome comes back as a system message.
  const runContextCommand = async (chatId, command, rawInput) => {
    if (command.type === 'open') {
      setShowContext(true);
      setInput('');
      return;
    }

    updateMessages(chatId, prev => [...prev, { role: 'user', content: rawInput.trim() }]);
    setChatLoading(chatId, true);
    setInput('');

    let result;
    let failure = null;
    try {
      if (command.type === 'compact') {
        const { data } = await axios.post(`${API_BASE}/chats/${chatId}/context/compact`, { focus: command.focus || '' });
        result = data.before === 0
          ? 'context is already empty'
          : `context compacted: ${data.before} ${data.before === 1 ? 'entry' : 'entries'} → ${data.after}`;
      } else if (command.type === 'keep') {
        const { data } = await axios.post(`${API_BASE}/chats/${chatId}/context/keep`, { topic: command.topic });
        if (data.before === 0) {
          result = 'context is already empty';
        } else if (data.kept === 0) {
          result = `nothing in the context was about "${command.topic}", so it is now empty`;
        } else {
          result = `context narrowed to "${command.topic}": ${data.before} ${data.before === 1 ? 'entry' : 'entries'} → ${data.after}`;
        }
      } else if (command.type === 'forget') {
        const { data } = await axios.post(`${API_BASE}/chats/${chatId}/context/forget`, { topic: command.topic });
        if (data.before === 0) {
          result = 'context is already empty';
        } else if (data.removed === 0) {
          result = `nothing in the context was about "${command.topic}"`;
        } else {
          result = `removed what the context said about "${command.topic}": ${data.before} ${data.before === 1 ? 'entry' : 'entries'} → ${data.after}`;
        }
      } else if (command.type === 'clear') {
        await axios.delete(`${API_BASE}/chats/${chatId}/context`);
        result = 'context cleared';
      } else {
        result = CONTEXT_USAGE;
      }
    } catch (error) {
      failure = readApiError(error);
    }

    updateMessages(chatId, prev => {
      const next = [...prev, failure
        ? { role: 'error', content: failure.message, code: failure.code }
        : { role: 'system', content: result }];
      persistMessages(chatId, next);
      return next;
    });
    setChatLoading(chatId, false);
    refreshContextUsage(chatId);
  };

  const runResearchCommand = async (chatId, command, rawInput) => {
    updateMessages(chatId, prev => [...prev, { role: 'user', content: rawInput.trim() }]);
    setInput('');

    let added;
    if (!command.topic) {
      added = { role: 'system', content: RESEARCH_USAGE };
    } else {
      try {
        const { data } = await axios.post(`${API_BASE}/chats/${chatId}/research`, { topic: command.topic });
        added = { role: 'research', research_id: data.id, topic: data.topic, content: '' };
      } catch (error) {
        added = { role: 'error', content: readApiError(error).message };
      }
    }
    updateMessages(chatId, prev => {
      const next = [...prev, added];
      persistMessages(chatId, next);
      return next;
    });
  };

  // The user's answer to "athena wants to run this code"
  const handlePermission = async (stepId, requestId, approve) => {
    const chatId = activeChatId;
    const setState = (state) => updateMessages(chatId, prev => prev.map(message => (
      message.steps ? {
        ...message,
        steps: message.steps.map(step => (
          step.id === stepId && step.permission?.id === requestId
            ? { ...step, permission: { ...step.permission, state } }
            : step
        )),
      } : message
    )));
    setState(approve ? 'approved' : 'denied');
    try {
      await axios.post(`${API_BASE}/chats/${chatId}/agent/permissions/${requestId}`, { approve });
    } catch (error) {
      setState('expired');
    }
  };

  const sendMessage = async (chatId, text) => {
    if (!chatId || !text.trim() || loadingByChat[chatId]) return;

    const command = parseContextCommand(text);
    if (command) {
      await runContextCommand(chatId, command, text);
      return;
    }

    const research = parseResearchCommand(text);
    if (research) {
      await runResearchCommand(chatId, research, text);
      return;
    }

    const currentInput = text;
    setInput('');

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
      // The server owns each chat's conversation context (editable via
      // /context), so only the new message is sent.
      const response = await fetch(`${API_BASE}/chats/${chatId}/qa/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: currentInput }),
        signal: controller.signal
      });

      if (!response.ok) {
        let detail;
        try { detail = (await response.json()).detail; } catch (_) { /* no readable body */ }
        throw new Error(typeof detail === 'string' && detail ? detail : `the server returned an error (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let completed = false;
      let buffer = '';

      // Updates the reply bubble that's currently streaming in
      const patchReply = (patch) => updateMessages(chatId, prev => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== 'assistant' || !last.streaming) return prev;
        const next = [...prev];
        next[next.length - 1] = typeof patch === 'function' ? patch(last) : { ...last, ...patch };
        return next;
      });
      const patchStep = (stepId, patch) => patchReply(message => ({
        ...message,
        steps: (message.steps || []).map(step => (step.id === stepId ? { ...step, ...patch } : step)),
      }));

      const handleEvent = (data) => {
        if (data.type === 'chunk' && data.content) {
          accumulatedContent += data.content;
          patchReply({ content: accumulatedContent });
        } else if (data.type === 'reset') {
          // What streamed so far was the agent thinking aloud before using a
          // tool, not the answer
          accumulatedContent = '';
          patchReply({ content: '' });
        } else if (data.type === 'step') {
          patchReply(message => ({
            ...message,
            steps: [...(message.steps || []), {
              id: data.id, tool: data.tool, args: data.args, thought: data.thought, status: 'running'
            }],
          }));
        } else if (data.type === 'step_result') {
          patchStep(data.id, { status: data.status, result: data.result });
        } else if (data.type === 'permission_request') {
          patchStep(data.step, { permission: { id: data.id, code: data.code, state: 'pending' } });
        } else if (data.type === 'notice') {
          patchReply({ notice: data.message });
        } else if (data.type === 'sources' && data.content) {
          patchReply({ sources: data.content });
        } else if (data.type === 'complete') {
          completed = true;
          updateMessages(chatId, prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === 'assistant' && lastMessage.streaming) {
              newMessages[newMessages.length - 1] = {
                ...lastMessage,
                streaming: false,
                response_time: data.response_time,
                content: accumulatedContent,
              };
            }
            persistMessages(chatId, newMessages);
            return newMessages;
          });
        } else if (data.type === 'error') {
          throw Object.assign(new Error(data.message), { errorCode: data.code });
        }
      };

      while (!completed) {
        const { done, value } = await reader.read();
        if (done) break;

        // A network chunk can end mid-line; hold the unfinished tail until
        // the rest of it arrives.
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ') || line.trim().length <= 6) continue;
          let data;
          try {
            data = JSON.parse(line.slice(6));
          } catch (parseError) {
            console.warn('Failed to parse SSE data:', line, parseError);
            continue;
          }
          // Not inside the try: the error event throws, and has to reach the
          // outer handler or the reply bubble is left blinking forever.
          handleEvent(data);
          if (completed) break;
        }
      }
      if (!completed) {
        throw new Error('the response ended before it finished. please try again.');
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        // Cancelled by the user - keep whatever content already streamed in,
        // just stop treating it as in-progress.
        updateMessages(chatId, prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage?.role === 'assistant' && lastMessage.streaming) {
            newMessages[newMessages.length - 1] = settleSteps({ ...lastMessage, streaming: false });
          }
          persistMessages(chatId, newMessages);
          return newMessages;
        });
      } else {
        const { message: failure, code } = readApiError(error);
        updateMessages(chatId, prev => {
          // Drop the empty bubble the reply was streaming into
          const newMessages = prev.slice(0, -1);
          const next = [...newMessages, { role: 'error', content: failure, code }];
          persistMessages(chatId, next);
          return next;
        });
      }
    }

    delete abortControllersRef.current[chatId];
    setChatLoading(chatId, false);
    refreshContextUsage(chatId);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    return sendMessage(activeChatId, input);
  };

  // Rewinds to before the last question, then either asks it again
  // ("regenerate") or puts it back in the box to change ("edit"). The model's
  // memory of that exchange is dropped too, so the old answer isn't counted
  // alongside the new one.
  const redoLastExchange = async (mode) => {
    const chatId = activeChatId;
    if (!chatId || loadingByChat[chatId]) return;
    const rewound = rewindLastExchange(messagesByChat[chatId] || []);
    if (!rewound) return;

    updateMessages(chatId, rewound.kept);
    persistMessages(chatId, rewound.kept);
    try {
      await axios.post(`${API_BASE}/chats/${chatId}/context/undo`, { question: rewound.text });
    } catch (error) {
      console.error('failed to undo the remembered exchange:', error);
    }

    if (mode === 'edit') {
      setInput(rewound.text);
      setTimeout(() => document.querySelector('.chat-input')?.focus(), 0);
      return;
    }
    await sendMessage(chatId, rewound.text);
  };

  const handleCancelResponse = () => {
    if (!activeChatId) return;
    abortControllersRef.current[activeChatId]?.abort();
  };

  const clearChat = () => {
    if (!activeChatId) return;
    updateMessages(activeChatId, []);
    persistMessages(activeChatId, []);
    // The model's context is what it remembers of this conversation, so
    // clearing the conversation clears that too.
    axios.delete(`${API_BASE}/chats/${activeChatId}/context`)
      .catch(error => console.error('failed to clear context:', error));
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
        ? 'reset timed out - please try again'
        : `reset failed: ${readApiError(error).message}`;
      updateMessages(chatId, prev => {
        const next = [...prev, { role: 'error', content: errorMsg }];
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

  const agentEnabled = !!chatConfig?.agent_enabled;
  const canChat = !!activeChatId && (dbStatus.exists || agentEnabled);

  const renderChatView = () => (
    <>
      <div className="title">document q&amp;a</div>
      <div className="subtitle">
        {agentEnabled
          ? 'ask anything · athena searches your documents and the web, calculates and runs code'
          : 'ask questions grounded in your uploaded pdfs'}
      </div>

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
            chatId={activeChatId}
            onDecidePermission={handlePermission}
            messages={messages}
            expandedSources={expandedSources}
            onToggleSources={toggleSources}
            onViewSource={setViewingSource}
            onOpenSettings={() => setShowApiKeyModal(true)}
            canRedo={!loading && canRedoLast(messages)}
            onRegenerate={() => redoLastExchange('regenerate')}
            onEdit={() => redoLastExchange('edit')}
          />
          <ChatInput
            input={input}
            setInput={setInput}
            onSubmit={handleSubmit}
            onCancel={handleCancelResponse}
            canChat={canChat}
            loading={loading}
            agentEnabled={agentEnabled}
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
          onOpenContext={() => setShowContext(true)}
          contextUsage={contextUsageByChat[activeChatId]}
          onOpenAgent={() => setShowAgent(true)}
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

        {showAgent && activeChatId && (
          <AgentModal
            chatId={activeChatId}
            onChanged={setChatConfig}
            onClose={() => setShowAgent(false)}
          />
        )}

        {showContext && activeChatId && (
          <ContextModal
            chatId={activeChatId}
            onChanged={() => refreshContextUsage(activeChatId)}
            onClose={() => setShowContext(false)}
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
