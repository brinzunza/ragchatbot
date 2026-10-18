import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';

const POLL_MS = 2000;

// The report marks headings and emphasis with **bold**
function renderBold(text) {
  return text.split(/(\*\*[^*]+?\*\*)/g).map((part, i) => (
    part.startsWith('**') && part.endsWith('**') && part.length > 4
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : part
  ));
}
const PHASE_LABEL = { planning: 'planning', researching: 'researching', writing: 'writing the report' };
const ENDED_LABEL = {
  done: 'finished',
  cancelled: 'cancelled',
  error: 'failed',
  interrupted: 'interrupted',
};

// A background research job: polls its progress, shows the plan and notes
// as they come in, and the report when it's written. The job lives on the
// server, so leaving and coming back (or reloading) picks it up again.
function ResearchCard({ chatId, researchId, topic }) {
  const [job, setJob] = useState(null);
  const [lost, setLost] = useState(false);
  const [showNotes, setShowNotes] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/chats/${chatId}/research/${researchId}`);
        if (cancelled) return;
        setJob(data);
        setLost(false);
        if (data.status === 'running') timer = setTimeout(poll, POLL_MS);
      } catch (error) {
        if (cancelled) return;
        if (error.response?.status === 404) {
          setLost(true);
        } else {
          timer = setTimeout(poll, POLL_MS * 2);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [chatId, researchId]);

  const cancel = async () => {
    try {
      await axios.post(`${API_BASE}/chats/${chatId}/research/${researchId}/cancel`);
    } catch (error) { /* it already finished */ }
  };

  const running = job?.status === 'running';

  return (
    <div className="research-card">
      <div className="research-head">
        <span className="research-label">research</span>
        <span className="research-status">
          {lost ? 'no longer available' : !job ? 'loading…' : running ? PHASE_LABEL[job.phase] : ENDED_LABEL[job.status]}
          {running && <span className="step-live-dot" aria-hidden="true" />}
        </span>
      </div>
      <div className="research-topic">{topic}</div>

      {job && job.plan.length > 0 && (
        <ol className="research-plan">
          {job.plan.map((question, i) => (
            <li key={i} className={i < job.notes.length ? 'research-plan-done' : ''}>{question}</li>
          ))}
        </ol>
      )}

      {running && job.current && <div className="research-current">{job.current}</div>}
      {running && (
        <button type="button" className="error-action" onClick={cancel}>cancel research</button>
      )}

      {job && (job.status === 'error' || job.status === 'interrupted') && (
        <div className="research-error">{job.error}</div>
      )}
      {job && job.status === 'cancelled' && job.notes.length > 0 && (
        <div className="research-current">stopped after {job.notes.length} of {job.plan.length} sub-questions</div>
      )}

      {job && job.report && <div className="research-report">{renderBold(job.report)}</div>}

      {job && job.notes.length > 0 && (
        <div className="research-notes">
          <button type="button" className="sources-toggle" onClick={() => setShowNotes(!showNotes)}>
            {showNotes ? 'hide' : 'show'} research notes ({job.notes.length})
          </button>
          {showNotes && job.notes.map((note, i) => (
            <div key={i} className="research-note">
              <div className="research-note-question">{note.question}</div>
              <div className="research-note-text">{renderBold(note.text)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ResearchCard;
