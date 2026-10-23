import React, { useEffect, useRef } from 'react';
import Message from './Message';

// How close to the bottom still counts as "following along"
const STICK_THRESHOLD_PX = 80;

function MessageList({
  chatId, onDecidePermission, messages, expandedSources, onToggleSources, onViewSource, onOpenSettings,
  canRedo, onRegenerate, onEdit
}) {
  const scroller = useRef(null);
  const content = useRef(null);
  const following = useRef(true);

  const scrollToBottom = () => {
    const element = scroller.current;
    if (element) element.scrollTop = element.scrollHeight;
  };

  // Follows new content (streamed text, tool steps, a research card filling in)
  // down the page - but only while the reader is at the bottom. Scroll up to
  // read something and it leaves you alone until you come back down.
  useEffect(() => {
    const element = scroller.current;
    const onScroll = () => {
      following.current = element.scrollHeight - element.scrollTop - element.clientHeight < STICK_THRESHOLD_PX;
    };
    element.addEventListener('scroll', onScroll, { passive: true });
    const observer = new ResizeObserver(() => {
      if (following.current) scrollToBottom();
    });
    observer.observe(content.current);
    return () => {
      element.removeEventListener('scroll', onScroll);
      observer.disconnect();
    };
  }, []);

  // A different chat, or the user sending something, always lands at the bottom
  useEffect(() => {
    following.current = true;
    scrollToBottom();
  }, [chatId]);

  const lastMessage = messages[messages.length - 1];
  const sentCount = messages.filter((message) => message.role === 'user').length;
  useEffect(() => {
    following.current = true;
    scrollToBottom();
  }, [sentCount]);

  const lastUserIndex = messages.reduce((found, message, i) => (message.role === 'user' ? i : found), -1);

  return (
    <div className="messages" ref={scroller}>
      <div className="messages-inner" ref={content}>
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
            isLastReply={message === lastMessage || index === lastUserIndex}
            canRedo={canRedo}
            onRegenerate={onRegenerate}
            onEdit={onEdit}
          />
        ))}
      </div>
    </div>
  );
}

export default MessageList;
