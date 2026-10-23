import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon } from './Icons';

const CLOUD = [
  { id: 'openai', name: 'OpenAI', placeholder: 'sk-...' },
  { id: 'anthropic', name: 'Claude', placeholder: 'sk-ant-...' },
];

function ApiKeyModal({ onClose, onChanged }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [drafts, setDrafts] = useState({ openai: '', anthropic: '' });
  const [shown, setShown] = useState({});
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [ollamaModels, setOllamaModels] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const response = await axios.get(`${API_BASE}/settings/api-key`);
        setStatus(response.data);
      } catch (err) {
        setError('failed to load api key status');
      }
      try {
        const response = await axios.get(`${API_BASE}/settings/ollama-models`);
        setOllamaModels(response.data.models);
      } catch (err) { /* the picker just stays hidden */ }
      setLoading(false);
    })();
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Runs one settings change; the backend answers with the new status.
  const run = async (label, request, fallbackError, after) => {
    setBusy(label);
    setError(null);
    try {
      const response = await request();
      setStatus(response.data);
      if (after) after();
      onChanged();
    } catch (err) {
      setError(err.response?.data?.detail || fallbackError);
    }
    setBusy(null);
  };

  const saveKey = (e, provider) => {
    e.preventDefault();
    const key = drafts[provider].trim();
    if (!key || busy) return;
    run(`save-${provider}`,
      () => axios.post(`${API_BASE}/settings/api-key`, { api_key: key, provider }),
      'failed to save api key',
      () => setDrafts((prev) => ({ ...prev, [provider]: '' })));
  };

  const removeKey = (provider) => run(`remove-${provider}`,
    () => axios.delete(`${API_BASE}/settings/api-key`, { params: { provider } }),
    'failed to remove api key');

  const switchProvider = (provider) => run(`use-${provider}`,
    () => axios.post(`${API_BASE}/settings/provider`, { provider }),
    'failed to switch provider');

  const chooseOllamaModel = (model) => run(`model-${model}`,
    () => axios.post(`${API_BASE}/settings/ollama-model`, { model }),
    'failed to switch model');

  const active = status?.provider_id;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>ai provider</h2>
          <button className="modal-close" onClick={onClose} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <div className="new-chat-form">
          {loading && <div className="modal-empty">loading...</div>}

          {!loading && status && (
            <div className="api-key-status">
              <span className={`status-dot ${status.has_key ? 'active' : ''}`} />
              using {status.provider} ({status.model})
            </div>
          )}

          {status && CLOUD.map(({ id, name, placeholder }) => {
            const info = status.providers?.[id] || { has_key: false, masked_key: null, model: '' };
            return (
              <form key={id} className="provider-card" onSubmit={(e) => saveKey(e, id)}>
                <div className="provider-head">
                  <span className="provider-name">{name}</span>
                  <span className="provider-model">{info.model}</span>
                  {active === id ? (
                    <span className="provider-active">in use</span>
                  ) : info.has_key && (
                    <button type="button" className="provider-use" disabled={!!busy} onClick={() => switchProvider(id)}>
                      use
                    </button>
                  )}
                </div>

                {info.has_key && (
                  <div className="provider-key">
                    key <code>{`${info.masked_key.slice(0, 4)}…${info.masked_key.slice(-4)}`}</code>
                    <button type="button" className="provider-remove" disabled={!!busy} onClick={() => removeKey(id)}>
                      {busy === `remove-${id}` ? 'removing...' : 'remove'}
                    </button>
                  </div>
                )}

                <div className="api-key-input-row">
                  <input
                    type={shown[id] ? 'text' : 'password'}
                    className="field-input"
                    placeholder={info.has_key ? `replace key (${placeholder})` : `${name} api key (${placeholder})`}
                    aria-label={`${name} api key`}
                    value={drafts[id]}
                    onChange={(e) => setDrafts((prev) => ({ ...prev, [id]: e.target.value }))}
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    type="button"
                    className="api-key-toggle"
                    onClick={() => setShown((prev) => ({ ...prev, [id]: !prev[id] }))}
                    aria-label={shown[id] ? 'hide key' : 'show key'}
                  >
                    {shown[id] ? 'hide' : 'show'}
                  </button>
                  <button type="submit" className="provider-use" disabled={!drafts[id].trim() || !!busy}>
                    {busy === `save-${id}` ? 'saving...' : 'save'}
                  </button>
                </div>
              </form>
            );
          })}

          {status && (
            <div className="provider-card">
              <div className="provider-head">
                <span className="provider-name">Ollama</span>
                {ollamaModels.length > 0 ? (
                  <select
                    className="provider-select"
                    aria-label="Ollama model"
                    value={status.providers.ollama?.model || ''}
                    disabled={!!busy}
                    onChange={(e) => chooseOllamaModel(e.target.value)}
                  >
                    {!ollamaModels.some((m) => m.name === status.providers.ollama?.model) && (
                      <option value={status.providers.ollama?.model}>{status.providers.ollama?.model}</option>
                    )}
                    {ollamaModels.map((m) => (
                      <option key={m.name} value={m.name}>{m.name} ({m.size_gb} GB)</option>
                    ))}
                  </select>
                ) : (
                  <span className="provider-model">{status.providers.ollama?.model || 'llama3.2:latest'}</span>
                )}
                {active === 'ollama' ? (
                  <span className="provider-active">in use</span>
                ) : (
                  <button type="button" className="provider-use" disabled={!!busy} onClick={() => switchProvider('ollama')}>
                    use
                  </button>
                )}
              </div>
            </div>
          )}

          {status && ollamaModels.length > 1 && (
            <div className="provider-hint">
              bigger local models follow instructions and use tools more reliably, but answer more slowly. documents you
              upload are summarized with the selected model too.
            </div>
          )}

          {error && <div className="field-error">{error}</div>}
        </div>
      </div>
    </div>
  );
}

export default ApiKeyModal;
