import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon } from './Icons';

function DocumentViewerModal({ name, startIndex, length, onClose }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const highlightRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    axios.get(`${API_BASE}/documents/${encodeURIComponent(name)}/content`)
      .then((response) => {
        if (!cancelled) setContent(response.data.content);
      })
      .catch(() => {
        if (!cancelled) setError('Failed to load document content');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [name]);

  useEffect(() => {
    // 'auto' (instant) rather than 'smooth' - smooth scrolling is an
    // animation, and animations are throttled or skipped entirely by the
    // browser in backgrounded/hidden tabs, which would silently no-op this.
    // useEffect already runs after layout is committed, so no rAF needed.
    highlightRef.current?.scrollIntoView({ block: 'center', behavior: 'auto' });
  }, [content]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const hasHighlight = typeof startIndex === 'number' && typeof length === 'number' && length > 0;
  const before = hasHighlight && content ? content.slice(0, startIndex) : content;
  const highlighted = hasHighlight && content ? content.slice(startIndex, startIndex + length) : '';
  const after = hasHighlight && content ? content.slice(startIndex + length) : '';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card viewer-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{name}</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </div>

        <div className="viewer-body">
          {loading && <div className="modal-empty">Loading...</div>}
          {!loading && error && <div className="modal-empty modal-error">{error}</div>}
          {!loading && !error && content != null && (
            <pre className="viewer-content">
              {before}
              {hasHighlight && (
                <mark ref={highlightRef}>{highlighted}</mark>
              )}
              {after}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentViewerModal;
