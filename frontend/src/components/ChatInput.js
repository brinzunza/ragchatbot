import React from 'react';
import { SendIcon, StopIcon } from './Icons';

function ChatInput({ input, setInput, onSubmit, onCancel, canChat, loading }) {
  return (
    <form onSubmit={onSubmit} className="chat-form">
      <div className="chat-input-wrap">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={canChat ? 'ask about your documents...' : 'upload pdf files first'}
          disabled={!canChat || loading}
          className="chat-input"
        />
        {loading ? (
          <button type="button" className="send-button stop-button" onClick={onCancel} aria-label="Stop generating">
            <StopIcon />
          </button>
        ) : (
          <button type="submit" className="send-button" disabled={!canChat || !input.trim()}>
            <SendIcon />
          </button>
        )}
      </div>
    </form>
  );
}

export default ChatInput;
