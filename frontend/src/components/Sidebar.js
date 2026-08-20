import React, { useState } from 'react';
import { TrashIcon } from './Icons';

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
      <div className="sidebar-brand">athena</div>

      <div className="sidebar-section">
        <div className="sidebar-label">chats</div>
        <div className="chat-list">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-list-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => onSwitchChat(chat.id)}
            >
              <span className="chat-list-name">{chat.name}</span>

              {confirmingChatId === chat.id ? (
                <span className="chat-list-confirm" onClick={(e) => e.stopPropagation()}>
                  <button className="confirm-yes" onClick={(e) => handleConfirmDelete(e, chat.id)}>yes</button>
                  <button className="confirm-cancel" onClick={(e) => { e.stopPropagation(); setConfirmingChatId(null); }}>no</button>
                </span>
              ) : (
                <button
                  className="chat-list-delete"
                  onClick={(e) => handleDeleteClick(e, chat.id)}
                  aria-label={`delete ${chat.name}`}
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          ))}
        </div>
        <button onClick={onNewChat} className="nav-item">
          new chat
        </button>
      </div>

      {activeChatId && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">knowledge base</div>
            <button onClick={onOpenLibrary} className="nav-item" disabled={loading}>
              documents
            </button>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">session</div>
            <button onClick={onClearChat} className="nav-item" disabled={loading}>
              clear chat
            </button>
            <button onClick={onResetDatabase} className="nav-item" disabled={loading}>
              reset database
            </button>
          </div>

          <div className="sidebar-status">
            <span className={`status-dot ${dbStatus.exists ? 'active' : ''}`} />
            {dbStatus.exists
              ? `${dbStatus.file_count} document${dbStatus.file_count === 1 ? '' : 's'} indexed`
              : 'no documents indexed'}
          </div>
        </>
      )}

      <div className="sidebar-section sidebar-section-settings">
        <div className="sidebar-label">help</div>
        <button onClick={onOpenInfo} className={`nav-item ${view === 'info' ? 'active' : ''}`}>
          how it works
        </button>
        <button onClick={onOpenSetup} className={`nav-item ${view === 'setup' ? 'active' : ''}`}>
          setup guide
        </button>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">settings</div>
        <button onClick={onOpenApiKey} className="nav-item">
          {apiKeyStatus && (
            <span className="nav-item-badge">
              {apiKeyStatus.has_key ? 'OpenAI' : 'Ollama'}
            </span>
          )}
          api key
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
