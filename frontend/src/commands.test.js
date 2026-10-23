import { parseContextCommand, parseResearchCommand } from './commands';

describe('parseContextCommand', () => {
  test.each([
    ['/context', { type: 'open' }],
    ['/CONTEXT', { type: 'open' }],
    ['  /context  ', { type: 'open' }],
    ['/context clear', { type: 'clear' }],
    ['/context compact', { type: 'compact', focus: '' }],
    ['/compact', { type: 'compact', focus: '' }],
    ['/compact keep the budget details', { type: 'compact', focus: 'the budget details' }],
    ['/compact focus on the dog', { type: 'compact', focus: 'the dog' }],
    ['/context compact focus on pricing', { type: 'compact', focus: 'pricing' }],
    ['/context keep only info on france', { type: 'keep', topic: 'france' }],
    ['/context keep information about taxes', { type: 'keep', topic: 'taxes' }],
    ['/context keep pricing', { type: 'keep', topic: 'pricing' }],
  ])('%s', (input, expected) => {
    expect(parseContextCommand(input)).toEqual(expected);
  });

  test.each([
    ['/forget john', 'john'],
    ['/forget the budget meeting', 'the budget meeting'],
    ['/compact remove all context about john', 'john'],
    ['/compact forget john', 'john'],
    ['/compact delete everything about the dog', 'the dog'],
    ['/compact drop anything on taxes', 'taxes'],
    ['/context forget the context about dogs', 'dogs'],
    ['/context remove info on Bob', 'Bob'],
    ['/context compact forget bob', 'bob'],
  ])('%s forgets "%s"', (input, topic) => {
    expect(parseContextCommand(input)).toEqual({ type: 'forget', topic });
  });

  test.each([
    ['/forget'],
    ['/context blah'],
    ['/context keep'],
    ['/context keep only'],
    ['/context keep info'],
  ])('%s is an unknown command (shows usage)', (input) => {
    expect(parseContextCommand(input)).toEqual({ type: 'unknown' });
  });

  test.each([
    ['hello'],
    ['what does /compact do?'],
    ['hello /compact'],
    ['/compactfoo'],
    ['/contextual'],
    ['/research solar'],
    [''],
  ])('%j is an ordinary message', (input) => {
    expect(parseContextCommand(input)).toBeNull();
  });
});

describe('parseResearchCommand', () => {
  test('extracts the topic', () => {
    expect(parseResearchCommand('/research heat pumps vs gas furnaces')).toEqual({ topic: 'heat pumps vs gas furnaces' });
    expect(parseResearchCommand('  /RESEARCH  solar  ')).toEqual({ topic: 'solar' });
  });

  test('a bare command has an empty topic (shows usage)', () => {
    expect(parseResearchCommand('/research')).toEqual({ topic: '' });
  });

  test('ordinary messages are not commands', () => {
    expect(parseResearchCommand('research heat pumps')).toBeNull();
    expect(parseResearchCommand('/researcher')).toBeNull();
    expect(parseResearchCommand('/context')).toBeNull();
  });
});
