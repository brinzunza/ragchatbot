import React from 'react';
import Message from './Message';

function MessageList({ messages, expandedSources, onToggleSources, messagesEndRef }) {
  return (
    <div className="messages">
      {messages.map((message, index) => (
        <Message
          key={index}
          message={message}
          expanded={!!expandedSources[index]}
          onToggleSources={() => onToggleSources(index)}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
