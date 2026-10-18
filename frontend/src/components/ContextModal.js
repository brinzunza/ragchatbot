import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon, TrashIcon } from './Icons';

const MIN_ENTRIES = 1;
const MAX_ENTRIES = 20;

function clip(text, limit) {
  return text.length <= limit ? text : `${text.slice(0, limit).trimEnd()}…`;
}

function ContextModal({ chatId, onClose }) {
  const [entries, setEntries] = useState([]);
  const [config, setConfig] = useState(null);
  const [maxDraft, setMaxDraft] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [confirmingClear, setConfirmingClear] = useState(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${API_BASE}/chats/${chatId}/context`);
        setEntries(response.data.entries);
        setConfig(response.data.config);
        setMaxDraft(String(response.data.config.max_context_entries));
      } catch (err) {
        setError('failed to load context');
      }
      setLoading(false);
    })();
  }, [chatId]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const updateConfig = async (patch) => {
    const previous = config;
    const next = { ...config, ...patch };
    setConfig(next);
    setError(null);
    try {
      await axios.put(`${API_BASE}/chats/${chatId}/config`, next);
    } catch (err) {
      setConfig(previous);
      setMaxDraft(String(previous.max_context_entries));
      setError('failed to save settings');
    }
  };

  const commitMaxDraft = () => {
    const value = parseInt(maxDraft, 10);
    if (Number.isNaN(value) || value < MIN_ENTRIES || value > MAX_ENTRIES) {
      setMaxDraft(String(config.max_context_entries));
      return;
    }
    if (value !== config.max_context_entries) updateConfig({ max_context_entries: value });
  };

  const handleDelete = async (id) => {
    setError(null);
    try {
      await axios.delete(`${API_BASE}/chats/${chatId}/context/${id}`);
      setEntries((prev) => prev.filter((entry) => entry.id !== id));
    } catch (err) {
      setError('failed to remove that entry');
    }
  };

  const handleClear = async () => {
    setError(null);
    try {
      await axios.delete(`${API_BASE}/chats/${chatId}/context`);
      setEntries([]);
    } catch (err) {
      setError('failed to clear context');
    }
    setConfirmingClear(false);
  };

  // Turns older than the window aren't sent to the model (the summary always is)
  const turns = entries.filter((entry) => entry.kind === 'turn');
  const outsideWindow = new Set(
    config ? turns.slice(0, Math.max(0, turns.length - config.max_context_entries)).map((entry) => entry.id) : []
  );

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>context</h2>
          <button className="modal-close" onClick={onClose} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <div className="modal-body">
          {loading && <div className="modal-empty">loading...</div>}

          {!loading && config && (
            <div className="context-config">
              <label className="radio-row">
                <input
                  type="checkbox"
                  checked={config.context_enabled}
                  onChange={(e) => updateConfig({ context_enabled: e.target.checked })}
                />
                include context in answers
              </label>
              <label className="context-config-row">
                recent exchanges sent to the model
                <input
                  type="number"
                  className="context-number"
                  min={MIN_ENTRIES}
                  max={MAX_ENTRIES}
                  value={maxDraft}
                  disabled={!config.context_enabled}
                  onChange={(e) => setMaxDraft(e.target.value)}
                  onBlur={commitMaxDraft}
                  onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                />
              </label>
            </div>
          )}

          {error && <div className="modal-empty modal-error">{error}</div>}

          {!loading && entries.length === 0 && (
            <div className="modal-empty">no context yet. it builds up as you chat.</div>
          )}

          {!loading && entries.map((entry) => (
            <div key={entry.id} className="document-row context-row">
              <div className="document-row-info">
                <div className="context-kind">
                  {entry.kind === 'summary' ? 'summary' : 'exchange'}
                  {outsideWindow.has(entry.id) && ' · outside window'}
                </div>
                {entry.kind === 'summary' ? (
                  <div className="context-answer context-summary">{entry.text}</div>
                ) : (
                  <>
                    <div className="document-row-name">{entry.question}</div>
                    <div className="context-answer" title={entry.answer}>{clip(entry.answer, 220)}</div>
                  </>
                )}
              </div>
              <button
                className="document-row-delete"
                onClick={() => handleDelete(entry.id)}
                aria-label="remove from context"
              >
                <TrashIcon />
              </button>
            </div>
          ))}

          {!loading && entries.length > 0 && (
            confirmingClear ? (
              <div className="document-row-confirm context-clear">
                <span>remove all context?</span>
                <button className="confirm-yes" onClick={handleClear}>yes</button>
                <button className="confirm-cancel" onClick={() => setConfirmingClear(false)}>cancel</button>
              </div>
            ) : (
              <button className="api-key-remove context-clear" onClick={() => setConfirmingClear(true)}>
                clear all
              </button>
            )
          )}

          <div className="context-hint">
            in the chat: <code>/compact</code> · <code>/forget &lt;topic&gt;</code> · <code>/context keep only info on …</code> · <code>/context clear</code>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ContextModal;
