// Slash commands typed into the chat box: /context, /compact, /forget and /research.
// Kept apart from App.js so they can be tested without rendering the app.

export const CONTEXT_USAGE = 'usage: /context (view and edit) · /compact [focus] · /forget <topic> · /context keep only info on <topic> · /context clear';

// "remove all context about john", "forget john", "drop anything on the budget"
const FORGET_PATTERN = /^(?:remove|forget|delete|drop|erase)\s+(?:all\s+|any\s+|everything\s+)?(?:(?:the\s+)?(?:context|info(?:rmation)?|memory|anything|mentions?|things?)\s+)?(?:about|on|regarding|of|related to)?\s*(.+)$/i;

// "focus on pricing" / "keeping pricing" -> "pricing"
const cleanFocus = (args) => args.replace(/^(?:focus(?:ing)?\s+on|keep(?:ing)?)\s+/i, '');

// Parses a context slash command - /context ..., plus the shorthands
// /compact [focus] and /forget <topic>; returns null for ordinary messages.
export function parseContextCommand(text) {
  const match = text.trim().match(/^\/(context|compact|forget)(?:\s+(.*))?$/i);
  if (!match) return null;
  const name = match[1].toLowerCase();
  const args = (match[2] || '').trim();

  if (name === 'forget') {
    return args ? { type: 'forget', topic: args } : { type: 'unknown' };
  }
  if (name === 'compact') {
    // "/compact remove all context about john" means what it says
    const forget = args.match(FORGET_PATTERN);
    if (forget) return { type: 'forget', topic: forget[1].trim() };
    return { type: 'compact', focus: cleanFocus(args) };
  }

  const rest = args;
  if (rest === '') return { type: 'open' };
  if (/^clear$/i.test(rest)) return { type: 'clear' };
  const compact = rest.match(/^compact(?:\s+(.*))?$/i);
  if (compact) {
    const focusArgs = (compact[1] || '').trim();
    const forgetFocus = focusArgs.match(FORGET_PATTERN);
    return forgetFocus
      ? { type: 'forget', topic: forgetFocus[1].trim() }
      : { type: 'compact', focus: cleanFocus(focusArgs) };
  }
  const forget = rest.match(FORGET_PATTERN);
  if (forget) return { type: 'forget', topic: forget[1].trim() };
  // "keep only info on X", "keep info about X", "keep X"
  const keep = rest.match(/^keep\s+(?:only\s+)?(?:info(?:rmation)?\s+)?(?:on\s+|about\s+)?(.+)$/i);
  if (keep && !/^(only|info|information|on|about)$/i.test(keep[1].trim())) {
    return { type: 'keep', topic: keep[1].trim() };
  }
  return { type: 'unknown' };
}

export const RESEARCH_USAGE = 'usage: /research <topic> · athena plans the topic, investigates it with your documents and the web in the background, and writes a report';

// Parses a "/research ..." slash command; returns null for ordinary messages.
export function parseResearchCommand(text) {
  const match = text.trim().match(/^\/research(?:\s+(.*))?$/i);
  return match ? { topic: (match[1] || '').trim() } : null;
}
