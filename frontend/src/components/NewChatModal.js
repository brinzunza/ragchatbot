import React, { useEffect, useState } from 'react';
import { CloseIcon } from './Icons';

function NewChatModal({ chats, onCreate, onClose }) {
  const [name, setName] = useState('');
  const [copyFrom, setCopyFrom] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    setError(null);
    try {
      await onCreate(name.trim(), copyFrom || null);
    } catch (err) {
      setError(err.response?.data?.detail || 'failed to create chat');
      setCreating(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>new chat</h2>
          <button className="modal-close" onClick={onClose} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <form className="new-chat-form" onSubmit={handleSubmit}>
          <label className="field-label" htmlFor="new-chat-name">name</label>
          <input
            id="new-chat-name"
            type="text"
            className="field-input"
            placeholder="e.g. research notes"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />

          <div className="field-label">documents</div>
          <label className="radio-row">
            <input
              type="radio"
              name="source"
              checked={copyFrom === ''}
              onChange={() => setCopyFrom('')}
            />
            start empty
          </label>
          <label className="radio-row">
            <input
              type="radio"
              name="source"
              checked={copyFrom !== ''}
              disabled={chats.length === 0}
              onChange={() => setCopyFrom(chats[0]?.id || '')}
            />
            copy from an existing chat
          </label>

          {copyFrom !== '' && (
            <select
              className="field-input"
              value={copyFrom}
              onChange={(e) => setCopyFrom(e.target.value)}
            >
              {chats.map((chat) => (
                <option key={chat.id} value={chat.id}>
                  {chat.name} ({chat.document_count} doc{chat.document_count === 1 ? '' : 's'})
                </option>
              ))}
            </select>
          )}

          {error && <div className="field-error">{error}</div>}

          <button type="submit" className="primary-action" disabled={!name.trim() || creating}>
            {creating ? 'creating...' : 'create chat'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default NewChatModal;
