import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon } from './Icons';

const MODES = [
  { value: 'safe', label: 'safe', hint: 'athena can search, calculate and browse, but never runs code.' },
  { value: 'ask', label: 'ask first', hint: 'athena shows you the code and waits for your OK before each run.' },
  { value: 'bypass', label: 'bypass', hint: 'code runs immediately, with no prompt.' },
];

const MIN_STEPS = 1;
const MAX_STEPS = 20;

function AgentModal({ chatId, onClose, onChanged }) {
  const [config, setConfig] = useState(null);
  const [stepsDraft, setStepsDraft] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/chats/${chatId}/config`);
        setConfig(data);
        setStepsDraft(String(data.max_agent_steps));
      } catch (err) {
        setError('failed to load settings');
      }
    })();
  }, [chatId]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const update = async (patch) => {
    const previous = config;
    const next = { ...config, ...patch };
    setConfig(next);
    setError(null);
    try {
      await axios.put(`${API_BASE}/chats/${chatId}/config`, next);
      onChanged?.(next);
    } catch (err) {
      setConfig(previous);
      setStepsDraft(String(previous.max_agent_steps));
      setError('failed to save settings');
    }
  };

  const commitSteps = () => {
    const value = parseInt(stepsDraft, 10);
    if (Number.isNaN(value) || value < MIN_STEPS || value > MAX_STEPS) {
      setStepsDraft(String(config.max_agent_steps));
      return;
    }
    if (value !== config.max_agent_steps) update({ max_agent_steps: value });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>agent</h2>
          <button className="modal-close" onClick={onClose} aria-label="close">
            <CloseIcon />
          </button>
        </div>

        <div className="modal-body">
          {!config && !error && <div className="modal-empty">loading...</div>}

          {config && (
            <>
              <div className="context-config">
                <label className="radio-row">
                  <input
                    type="checkbox"
                    checked={config.agent_enabled}
                    onChange={(e) => update({ agent_enabled: e.target.checked })}
                  />
                  let athena decide which tools to use
                </label>
                <label className="radio-row">
                  <input
                    type="checkbox"
                    checked={config.web_search_enabled}
                    disabled={!config.agent_enabled}
                    onChange={(e) => update({ web_search_enabled: e.target.checked })}
                  />
                  allow web search
                </label>
                <label className="context-config-row">
                  most tool calls per message
                  <input
                    type="number"
                    className="context-number"
                    min={MIN_STEPS}
                    max={MAX_STEPS}
                    value={stepsDraft}
                    disabled={!config.agent_enabled}
                    onChange={(e) => setStepsDraft(e.target.value)}
                    onBlur={commitSteps}
                    onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                  />
                </label>
              </div>

              <div className="agent-modes">
                <div className="sidebar-label">running code</div>
                {MODES.map((mode) => (
                  <label key={mode.value} className="agent-mode">
                    <input
                      type="radio"
                      name="permission_mode"
                      checked={config.permission_mode === mode.value}
                      disabled={!config.agent_enabled}
                      onChange={() => update({ permission_mode: mode.value })}
                    />
                    <span>
                      <span className="agent-mode-name">{mode.label}</span>
                      <span className="agent-mode-hint">{mode.hint}</span>
                    </span>
                  </label>
                ))}
                {config.permission_mode === 'bypass' && config.agent_enabled && (
                  <div className="agent-warning">
                    bypass lets model-written code run unattended. it runs in this chat's workspace folder with
                    a time limit, but it is not a sandbox: it can read files and use the network like any
                    program you run. background research runs code only in this mode.
                  </div>
                )}
              </div>
            </>
          )}

          {error && <div className="modal-empty modal-error">{error}</div>}

          <div className="context-hint">
            <code>/research &lt;topic&gt;</code> starts a long background investigation of a topic.
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentModal;
