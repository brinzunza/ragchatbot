import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE } from '../App';
import { CloseIcon, SearchIcon, TrashIcon } from './Icons';

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function DocumentLibraryModal({ chatId, onClose, onDocumentDeleted }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [confirmingName, setConfirmingName] = useState(null);
  const [deletingName, setDeletingName] = useState(null);

  useEffect(() => {
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE}/chats/${chatId}/documents`);
      setDocuments(response.data);
    } catch (err) {
      setError('Failed to load documents');
    }
    setLoading(false);
  };

  const handleDelete = async (name) => {
    setDeletingName(name);
    try {
      await axios.delete(`${API_BASE}/chats/${chatId}/documents/${encodeURIComponent(name)}`);
      setDocuments((prev) => prev.filter((doc) => doc.name !== name));
      onDocumentDeleted();
    } catch (err) {
      setError(`Failed to delete "${name}"`);
    }
    setConfirmingName(null);
    setDeletingName(null);
  };

  const filtered = documents.filter((doc) =>
    doc.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Documents</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </div>

        <div className="modal-search">
          <SearchIcon />
          <input
            type="text"
            placeholder="Search documents..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
        </div>

        <div className="modal-body">
          {loading && <div className="modal-empty">Loading...</div>}

          {!loading && error && <div className="modal-empty modal-error">{error}</div>}

          {!loading && !error && documents.length === 0 && (
            <div className="modal-empty">No documents indexed yet.</div>
          )}

          {!loading && !error && documents.length > 0 && filtered.length === 0 && (
            <div className="modal-empty">No documents match "{query}".</div>
          )}

          {!loading && !error && filtered.map((doc) => (
            <div key={doc.name} className="document-row">
              <div className="document-row-info">
                <div className="document-row-name">{doc.name}</div>
                <div className="document-row-meta">
                  {doc.chunk_count} chunk{doc.chunk_count === 1 ? '' : 's'}
                  {doc.uploaded_at && ` · ${formatDate(doc.uploaded_at)}`}
                </div>
              </div>

              {confirmingName === doc.name ? (
                <div className="document-row-confirm">
                  <span>Delete?</span>
                  <button
                    className="confirm-yes"
                    onClick={() => handleDelete(doc.name)}
                    disabled={deletingName === doc.name}
                  >
                    {deletingName === doc.name ? 'Deleting...' : 'Yes'}
                  </button>
                  <button className="confirm-cancel" onClick={() => setConfirmingName(null)}>
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  className="document-row-delete"
                  onClick={() => setConfirmingName(doc.name)}
                  aria-label={`Delete ${doc.name}`}
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default DocumentLibraryModal;
