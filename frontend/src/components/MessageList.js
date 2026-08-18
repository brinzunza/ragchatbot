import React from 'react';
import Message from './Message';

function MessageList({ messages, expandedSources, onToggleSources, onViewSource, messagesEndRef }) {
  return (
    <div className="messages">
      {messages.map((message, index) => (
        <Message
          key={index}
          message={message}
          expanded={!!expandedSources[index]}
          onToggleSources={() => onToggleSources(index)}
          onViewSource={onViewSource}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
