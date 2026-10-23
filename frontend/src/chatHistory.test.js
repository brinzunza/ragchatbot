import { canRedoLast, rewindLastExchange } from './chatHistory';

const user = (content) => ({ role: 'user', content });
const assistant = (content, extra = {}) => ({ role: 'assistant', content, ...extra });

describe('rewindLastExchange', () => {
  test('drops the last question and everything after it', () => {
    const messages = [user('q1'), assistant('a1'), user('q2'), assistant('a2', { steps: [{ id: 1 }] })];
    expect(rewindLastExchange(messages)).toEqual({ kept: messages.slice(0, 2), text: 'q2' });
  });

  test('also drops errors and notes that followed the question', () => {
    const messages = [user('q1'), assistant('a1'), user('q2'), { role: 'error', content: 'boom' }];
    expect(rewindLastExchange(messages).kept).toEqual([user('q1'), assistant('a1')]);
  });

  test('a lone question rewinds to an empty conversation', () => {
    expect(rewindLastExchange([user('q1')])).toEqual({ kept: [], text: 'q1' });
  });

  test('is null when nobody has asked anything', () => {
    expect(rewindLastExchange([])).toBeNull();
    expect(rewindLastExchange([{ role: 'system', content: 'Database reset' }])).toBeNull();
  });

  test('does not mutate its input', () => {
    const messages = [user('q1'), assistant('a1')];
    rewindLastExchange(messages);
    expect(messages).toHaveLength(2);
  });
});

describe('canRedoLast', () => {
  test('is true after a finished reply or an error', () => {
    expect(canRedoLast([user('q'), assistant('a')])).toBe(true);
    expect(canRedoLast([user('q'), { role: 'error', content: 'x' }])).toBe(true);
  });

  test('is false while a reply is still streaming', () => {
    expect(canRedoLast([user('q'), assistant('par', { streaming: true })])).toBe(false);
  });

  test('is false for an empty chat, a note, or a research card at the end', () => {
    expect(canRedoLast([])).toBe(false);
    expect(canRedoLast([user('q'), { role: 'system', content: 'context cleared' }])).toBe(false);
    expect(canRedoLast([user('/research x'), { role: 'research', research_id: 'abc' }])).toBe(false);
  });

  test('is false when there is no question to redo', () => {
    expect(canRedoLast([assistant('stray')])).toBe(false);
  });
});
