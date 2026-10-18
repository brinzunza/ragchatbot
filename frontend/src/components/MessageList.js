import React from 'react';
import Message from './Message';

function MessageList({ chatId, onDecidePermission, messages, expandedSources, onToggleSources, onViewSource, onOpenSettings, messagesEndRef }) {
  return (
    <div className="messages">
      {messages.map((message, index) => (
        <Message
          key={index}
          chatId={chatId}
          message={message}
          onDecidePermission={onDecidePermission}
          expanded={!!expandedSources[index]}
          onToggleSources={() => onToggleSources(index)}
          onViewSource={onViewSource}
          onOpenSettings={onOpenSettings}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
