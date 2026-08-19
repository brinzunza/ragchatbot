import React, { useState } from 'react';
import { TrashIcon, RefreshIcon, FileTextIcon, PlusIcon, MessageIcon, KeyIcon, InfoIcon, CompassIcon } from './Icons';

function Sidebar({
  chats,
  activeChatId,
  onSwitchChat,
  onNewChat,
  onDeleteChat,
  onClearChat,
  onResetDatabase,
  onOpenLibrary,
  onOpenApiKey,
  apiKeyStatus,
  view,
  onOpenInfo,
  onOpenSetup,
  loading,
  dbStatus
}) {
  const [confirmingChatId, setConfirmingChatId] = useState(null);

  const handleDeleteClick = (e, chatId) => {
    e.stopPropagation();
    setConfirmingChatId(chatId);
  };

  const handleConfirmDelete = (e, chatId) => {
    e.stopPropagation();
    setConfirmingChatId(null);
    onDeleteChat(chatId);
  };

  return (
    <div className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark">A</span>
        Athena
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Chats</div>
        <div className="chat-list">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-list-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => onSwitchChat(chat.id)}
            >
              <MessageIcon />
              <span className="chat-list-name">{chat.name}</span>

              {confirmingChatId === chat.id ? (
                <span className="chat-list-confirm" onClick={(e) => e.stopPropagation()}>
                  <button className="confirm-yes" onClick={(e) => handleConfirmDelete(e, chat.id)}>Yes</button>
                  <button className="confirm-cancel" onClick={(e) => { e.stopPropagation(); setConfirmingChatId(null); }}>No</button>
                </span>
              ) : (
                <button
                  className="chat-list-delete"
                  onClick={(e) => handleDeleteClick(e, chat.id)}
                  aria-label={`Delete ${chat.name}`}
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          ))}
        </div>
        <button onClick={onNewChat} className="nav-item">
          <PlusIcon />
          New chat
        </button>
      </div>

      {activeChatId && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">Knowledge Base</div>
            <button onClick={onOpenLibrary} className="nav-item" disabled={loading}>
              <FileTextIcon />
              Documents
            </button>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">Session</div>
            <button onClick={onClearChat} className="nav-item" disabled={loading}>
              <TrashIcon />
              Clear chat
            </button>
            <button onClick={onResetDatabase} className="nav-item" disabled={loading}>
              <RefreshIcon />
              Reset database
            </button>
          </div>

          <div className="sidebar-status">
            <span className={`status-dot ${dbStatus.exists ? 'active' : ''}`} />
            {dbStatus.exists
              ? `${dbStatus.file_count} document${dbStatus.file_count === 1 ? '' : 's'} indexed`
              : 'No documents indexed'}
          </div>
        </>
      )}

      <div className="sidebar-section sidebar-section-settings">
        <div className="sidebar-label">Help</div>
        <button onClick={onOpenInfo} className={`nav-item ${view === 'info' ? 'active' : ''}`}>
          <InfoIcon />
          How it works
        </button>
        <button onClick={onOpenSetup} className={`nav-item ${view === 'setup' ? 'active' : ''}`}>
          <CompassIcon />
          Setup guide
        </button>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Settings</div>
        <button onClick={onOpenApiKey} className="nav-item">
          <KeyIcon />
          API key
          {apiKeyStatus && (
            <span className="nav-item-badge">
              {apiKeyStatus.has_key ? 'OpenAI' : 'Ollama'}
            </span>
          )}
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
