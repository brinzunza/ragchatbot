import React, { useState } from 'react';
import { ChevronRightIcon } from './Icons';

const TOOL_LABELS = {
  search_documents: 'searched documents',
  read_document: 'read document',
  list_documents: 'listed documents',
  web_search: 'searched the web',
  fetch_url: 'read web page',
  calculator: 'calculated',
  current_datetime: 'checked the date',
  run_python: 'ran python',
  compact_context: 'compacted context',
  keep_only_context: 'narrowed context',
  remember: 'remembered',
};

const STATUS_MARK = {
  running: 'working…',
  ok: 'done',
  error: 'failed',
  denied: 'not allowed',
  stopped: 'stopped',
  skipped: 'redirected',
};

function clip(text, limit) {
  const value = String(text);
  return value.length <= limit ? value : `${value.slice(0, limit).trimEnd()}…`;
}

// The one argument worth showing next to the tool's name
function summarize(step) {
  const args = step.args || {};
  if (step.tool === 'run_python') return '';
  const value = args.query ?? args.expression ?? args.url ?? args.name ?? args.topic ?? args.fact;
  return value ? clip(value, 90) : '';
}

function PermissionPrompt({ permission, onDecide }) {
  const decided = permission.state !== 'pending';
  return (
    <div className="permission-prompt">
      <div className="permission-title">
        {permission.state === 'pending' && 'athena wants to run this code on your computer'}
        {permission.state === 'approved' && 'you allowed this code to run'}
        {permission.state === 'denied' && 'you did not allow this code to run'}
        {permission.state === 'expired' && 'this request expired'}
      </div>
      <pre className="step-code">{permission.code}</pre>
      {!decided && (
        <div className="permission-actions">
          <button type="button" className="error-action" onClick={() => onDecide(true)}>allow once</button>
          <button type="button" className="error-action" onClick={() => onDecide(false)}>don't allow</button>
        </div>
      )}
    </div>
  );
}

function AgentSteps({ steps, streaming, onDecide }) {
  const [openChoice, setOpenChoice] = useState(null);
  const awaitingApproval = steps.some((s) => s.permission?.state === 'pending');
  // Follows the run by default (open while working, tucked away after);
  // once the user toggles it, their choice sticks - except that a pending
  // approval always shows, since the agent is stuck until it's answered.
  const open = awaitingApproval || (openChoice ?? streaming);
  const running = steps.some((s) => s.status === 'running');

  return (
    <div className="agent-steps">
      <button
        type="button"
        className="sources-toggle"
        onClick={() => setOpenChoice(!open)}
        aria-expanded={open}
      >
        <span className={`arrow ${open ? 'expanded' : ''}`}>
          <ChevronRightIcon />
        </span>
        {steps.length} {steps.length === 1 ? 'step' : 'steps'}
        {running && <span className="step-live-dot" aria-hidden="true" />}
      </button>
      {open && (
        <ol className="step-list">
          {steps.map((step) => (
            <li key={step.id} className={`step step-${step.status}`}>
              {step.thought && <div className="step-thought">{clip(step.thought, 240)}</div>}
              <div className="step-head">
                <span className="step-tool">{TOOL_LABELS[step.tool] || step.tool}</span>
                <span className="step-arg">{summarize(step)}</span>
                <span className="step-status">{STATUS_MARK[step.status] || step.status}</span>
              </div>
              {step.tool === 'run_python' && !step.permission && step.args?.code && (
                <pre className="step-code">{step.args.code}</pre>
              )}
              {step.permission && (
                <PermissionPrompt
                  permission={step.permission}
                  onDecide={(approve) => onDecide(step.id, step.permission.id, approve)}
                />
              )}
              {step.result && <div className="step-result">{clip(step.result, 400)}</div>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default AgentSteps;
