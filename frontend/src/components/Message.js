import React from 'react';
import { ChevronRightIcon } from './Icons';

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

function Message({ message, expanded, onToggleSources }) {
  return (
    <div className={`message ${message.role}`}>
      <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
        {renderContent(message.content)}
        {message.response_time != null && (
          <div className="response-time">{message.response_time.toFixed(2)}s</div>
        )}
      </div>
      {message.sources && message.role === 'assistant' && (
        <div className="sources-container">
          <button className="sources-toggle" onClick={onToggleSources}>
            <span className={`arrow ${expanded ? 'expanded' : ''}`}>
              <ChevronRightIcon />
            </span>
            Sources
          </button>
          {expanded && <div className="sources-content">{message.sources}</div>}
        </div>
      )}
    </div>
  );
}

export default Message;
