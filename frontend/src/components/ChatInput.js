import React from 'react';
import { SendIcon } from './Icons';

function ChatInput({ input, setInput, onSubmit, canChat, loading }) {
  return (
    <form onSubmit={onSubmit} className="chat-form">
      <div className="chat-input-wrap">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={canChat ? 'Ask about your documents...' : 'Upload PDF files first'}
          disabled={!canChat || loading}
          className="chat-input"
        />
        <button type="submit" className="send-button" disabled={!canChat || loading || !input.trim()}>
          <SendIcon />
        </button>
      </div>
    </form>
  );
}

export default ChatInput;
