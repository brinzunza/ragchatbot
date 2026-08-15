import React from 'react';
import { TrashIcon, RefreshIcon } from './Icons';

function Sidebar({ onClearChat, onResetDatabase, loading, dbStatus }) {
  return (
    <div className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark">A</span>
        Athena
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Session</div>
        <button onClick={onClearChat} className="nav-item" disabled={loading}>
          <TrashIcon />
          Clear chat
        </button>
        <button onClick={onResetDatabase} className="nav-item" disabled={loading}>
          <RefreshIcon />
          Reset database
        </button>
      </div>

      <div className="sidebar-status">
        <span className={`status-dot ${dbStatus.exists ? 'active' : ''}`} />
        {dbStatus.exists
          ? `${dbStatus.file_count} document${dbStatus.file_count === 1 ? '' : 's'} indexed`
          : 'No documents indexed'}
      </div>
    </div>
  );
}

export default Sidebar;
