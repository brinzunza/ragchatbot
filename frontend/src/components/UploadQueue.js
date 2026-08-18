import React from 'react';

function UploadQueue({ uploadQueue }) {
  if (uploadQueue.length === 0) return null;

  return (
    <div className="upload-queue">
      <div className="queue-header">Uploading ({uploadQueue.length})</div>
      {uploadQueue.map((upload) => {
        const active = upload.status !== 'completed' && upload.status !== 'error';
        return (
        <div key={upload.uploadId} className="upload-progress">
          <div className="progress-info">
            <span className="progress-message">
              {active && <span className="progress-live-dot" />}
              {upload.message}
            </span>
            <span className="progress-percentage">{upload.progress}%</span>
          </div>
          <div className="progress-bar-container">
            <div
              className={`progress-bar-fill ${active ? 'active' : ''}`}
              style={{ width: `${upload.progress}%` }}
            />
          </div>
          {upload.currentFile && (
            <div className="progress-file">Processing: {upload.currentFile}</div>
          )}
          {upload.totalFiles > 0 && (
            <div className="progress-count">{upload.processedFiles} / {upload.totalFiles} files</div>
          )}
          {upload.status === 'uploading' && !upload.currentFile && (
            <div className="progress-file">{upload.fileNames}</div>
          )}
        </div>
        );
      })}
    </div>
  );
}

export default UploadQueue;
