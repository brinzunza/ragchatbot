import React from 'react';
import { ChevronRightIcon, FileTextIcon } from './Icons';
import AgentSteps from './AgentSteps';
import ResearchCard from './ResearchCard';

// Splits on **bold** spans (Knowledge-Base-sourced content, per the generator's system prompt)
// and renders them as <strong>. Everything else renders as plain text.
function renderContent(text) {
  const parts = text.split(/(\*\*[^*]+?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

// Error codes (from the backend) that the API key settings can fix
const SETTINGS_ERROR_CODES = new Set(['no_credits', 'invalid_key', 'ollama_unreachable']);

function Message({ chatId, message, expanded, onToggleSources, onViewSource, onOpenSettings, onDecidePermission }) {
  const sources = message.sources || [];

  if (message.role === 'error') {
    return (
      <div className="message error" role="alert">
        <div className="message-content">
          <div className="error-label">error</div>
          {message.content}
          {onOpenSettings && SETTINGS_ERROR_CODES.has(message.code) && (
            <button type="button" className="error-action" onClick={onOpenSettings}>
              open provider settings
            </button>
          )}
        </div>
      </div>
    );
  }

  if (message.role === 'research') {
    return (
      <div className="message research">
        <ResearchCard chatId={chatId} researchId={message.research_id} topic={message.topic} />
      </div>
    );
  }

  const steps = message.role === 'assistant' ? message.steps || [] : [];
  const hasContent = message.content || !message.streaming || steps.length === 0;

  return (
    <div className={`message ${message.role}`}>
      {steps.length > 0 && (
        <AgentSteps steps={steps} streaming={!!message.streaming} onDecide={onDecidePermission} />
      )}
      {message.notice && <div className="message-notice">{message.notice}</div>}
      {hasContent && <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
        {renderContent(message.content)}
        {message.response_time != null && (
          <div className="response-time">{message.response_time.toFixed(2)}s</div>
        )}
      </div>}
      {sources.length > 0 && message.role === 'assistant' && (
        <div className="sources-container">
          <button className="sources-toggle" onClick={onToggleSources}>
            <span className={`arrow ${expanded ? 'expanded' : ''}`}>
              <ChevronRightIcon />
            </span>
            {sources.length === 1 ? 'source' : `sources (${sources.length})`}
          </button>
          {expanded && (
            <div className="source-chips">
              {sources.map((source, i) => (
                <button
                  key={i}
                  className="source-chip"
                  onClick={() => onViewSource(source)}
                  title={`View the passage cited from ${source.name}`}
                >
                  <FileTextIcon />
                  {source.name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Message;
