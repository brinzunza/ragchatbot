// Helpers for rewinding a conversation to before its last question, which is
// what "regenerate" and "edit" both start from.

// Everything from the last user message onward belongs to that exchange (the
// reply, or an error, or a command's result). Returns the messages before it
// and the user's text, or null if there is no user message to rewind to.
export function rewindLastExchange(messages) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') {
      return { kept: messages.slice(0, i), text: messages[i].content };
    }
  }
  return null;
}

// Whether the last exchange can be redone: the conversation ends in a
// finished reply (or an error) to a user message, not in a note or a
// background research card.
export function canRedoLast(messages) {
  const last = messages[messages.length - 1];
  if (!last || last.streaming) return false;
  if (last.role !== 'assistant' && last.role !== 'error') return false;
  return messages.some((message) => message.role === 'user');
}
