import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon } from './Icons';

function ApiKeyModal({ onClose, onChanged }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/settings/api-key`);
      setStatus(response.data);
    } catch (err) {
      setError('Failed to load API key status');
    }
    setLoading(false);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!apiKey.trim() || saving) return;
    setSaving(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/settings/api-key`, { api_key: apiKey.trim() });
      setStatus(response.data);
      setApiKey('');
      onChanged();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to save API key');
    }
    setSaving(false);
  };

  const handleRemove = async () => {
    setRemoving(true);
    setError(null);
    try {
      const response = await axios.delete(`${API_BASE}/settings/api-key`);
      setStatus(response.data);
      onChanged();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to remove API key');
    }
    setRemoving(false);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>OpenAI API key</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </div>

        <div className="new-chat-form">
          {loading && <div className="modal-empty">Loading...</div>}

          {!loading && status && (
            <div className="api-key-status">
              {status.has_key ? (
                <>
                  <span className={`status-dot active`} />
                  Using OpenAI ({status.model}) — key <code>{status.masked_key}</code>
                </>
              ) : (
                <>
                  <span className="status-dot" />
                  No key set — using local Ollama ({status.model})
                </>
              )}
            </div>
          )}

          <form onSubmit={handleSave}>
            <label className="field-label" htmlFor="api-key-input">
              {status?.has_key ? 'Replace key' : 'Add a key'}
            </label>
            <div className="api-key-input-row">
              <input
                id="api-key-input"
                type={showKey ? 'text' : 'password'}
                className="field-input"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                autoFocus
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className="api-key-toggle"
                onClick={() => setShowKey((v) => !v)}
                aria-label={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>

            {error && <div className="field-error">{error}</div>}

            <button type="submit" className="primary-action" disabled={!apiKey.trim() || saving}>
              {saving ? 'Saving...' : 'Save key'}
            </button>
          </form>

          {status?.has_key && (
            <button
              type="button"
              className="api-key-remove"
              onClick={handleRemove}
              disabled={removing}
            >
              {removing ? 'Removing...' : 'Remove key and use local Ollama instead'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default ApiKeyModal;
