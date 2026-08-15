import React from 'react';
import { ChevronRightIcon } from './Icons';

function Message({ message, expanded, onToggleSources }) {
  return (
    <div className={`message ${message.role}`}>
      <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
        {message.content}
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
