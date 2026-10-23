import React, { useState } from 'react';

function Sidebar({
  chats,
  activeChatId,
  onSwitchChat,
  onNewChat,
  onDeleteChat,
  onClearChat,
  onResetDatabase,
  onOpenLibrary,
  onOpenContext,
  contextUsage,
  onOpenAgent,
  onOpenApiKey,
  apiKeyStatus,
  view,
  onOpenInfo,
  onOpenSetup,
  loading,
  dbStatus
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const handleConfirmDelete = () => {
    setConfirmingDelete(false);
    onDeleteChat(activeChatId);
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
            <button onClick={onOpenAgent} className="nav-item">
              agent
            </button>
            <button onClick={onOpenContext} className="nav-item">
              {contextUsage?.enabled && contextUsage.total_exchanges > 0 && (
                <span
                  className={`nav-item-badge ${contextUsage.left_out > 0 ? 'nav-item-badge-warn' : ''}`}
                  title={contextUsage.left_out > 0
                    ? `${contextUsage.left_out} older exchanges are no longer sent to the model`
                    : 'how much of the context window each message uses'}
                >
                  {contextUsage.percent}%
                </span>
              )}
              context
            </button>
            <button onClick={onClearChat} className="nav-item" disabled={loading}>
              clear chat
            </button>
            <button onClick={onResetDatabase} className="nav-item" disabled={loading}>
              reset database
            </button>
            {confirmingDelete ? (
              <span className="chat-list-confirm session-delete-confirm">
                delete this chat?
                <button className="confirm-yes" onClick={handleConfirmDelete}>yes</button>
                <button className="confirm-cancel" onClick={() => setConfirmingDelete(false)}>no</button>
              </span>
            ) : (
              <button onClick={() => setConfirmingDelete(true)} className="nav-item" disabled={loading}>
                delete chat
              </button>
            )}
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
              {apiKeyStatus.provider}
            </span>
          )}
          ai provider
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
