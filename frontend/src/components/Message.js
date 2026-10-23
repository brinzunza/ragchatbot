import React, { useState } from 'react';
import { ChevronRightIcon, FileTextIcon } from './Icons';
import AgentSteps from './AgentSteps';
import ResearchCard from './ResearchCard';
import Markdown from './Markdown';

// Error codes (from the backend) that the API key settings can fix
const SETTINGS_ERROR_CODES = new Set(['no_credits', 'invalid_key', 'ollama_unreachable']);

// Small text buttons under a message: copy, and (on the last exchange) redo it
function MessageActions({ children }) {
  return <div className="message-actions">{children}</div>;
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (error) {
      console.error('copy failed:', error);
    }
  };
  return <button type="button" className="message-action" onClick={copy}>{copied ? 'copied' : 'copy'}</button>;
}

function Message({
  chatId, message, expanded, onToggleSources, onViewSource, onOpenSettings, onDecidePermission,
  isLastReply, canRedo, onRegenerate, onEdit
}) {
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
          {isLastReply && canRedo && (
            <button type="button" className="error-action" onClick={onRegenerate}>
              try again
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
  const finished = message.role === 'assistant' && !message.streaming && message.content;

  return (
    <div className={`message ${message.role}`}>
      {steps.length > 0 && (
        <AgentSteps steps={steps} streaming={!!message.streaming} onDecide={onDecidePermission} />
      )}
      {message.notice && <div className="message-notice">{message.notice}</div>}
      {hasContent && <div className={`message-content ${message.streaming ? 'streaming' : ''}`}>
        {message.role === 'assistant' ? <Markdown>{message.content}</Markdown> : message.content}
        {message.response_time != null && (
          <div className="response-time">{message.response_time.toFixed(2)}s</div>
        )}
      </div>}
      {finished && (
        <MessageActions>
          <CopyButton text={message.content} />
          {isLastReply && canRedo && (
            <button type="button" className="message-action" onClick={onRegenerate}>regenerate</button>
          )}
        </MessageActions>
      )}
      {message.role === 'user' && isLastReply && canRedo && (
        <MessageActions>
          <button type="button" className="message-action" onClick={onEdit}>edit</button>
        </MessageActions>
      )}
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
