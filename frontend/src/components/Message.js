import React from 'react';
import { ChevronRightIcon, FileTextIcon } from './Icons';

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

function Message({ message, expanded, onToggleSources, onViewSource }) {
  const sources = message.sources || [];

  return (
    <div className={`message ${message.role}`}>
      <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
        {renderContent(message.content)}
        {message.response_time != null && (
          <div className="response-time">{message.response_time.toFixed(2)}s</div>
        )}
      </div>
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
