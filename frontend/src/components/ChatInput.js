import React from 'react';
import { SendIcon, StopIcon } from './Icons';

function ChatInput({ input, setInput, onSubmit, onCancel, canChat, loading, agentEnabled }) {
  return (
    <form onSubmit={onSubmit} className="chat-form">
      <div className="chat-input-wrap">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={canChat ? (agentEnabled ? 'ask anything...' : 'ask about your documents...') : 'upload pdf files first'}
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
      {input.startsWith('/') && (
        <div className="command-hint">
          <code>/context</code> view and edit · <code>/compact</code> · <code>/forget &lt;topic&gt;</code> · <code>/context keep only info on …</code> · <code>/context clear</code> · <code>/research &lt;topic&gt;</code>
        </div>
      )}
    </form>
  );
}

export default ChatInput;
