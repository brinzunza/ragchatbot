import React, { useRef, useState } from 'react';
import { UploadCloudIcon } from './Icons';

function UploadDropzone({ onFiles, disabled, compact }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (disabled) return;
    if (e.dataTransfer.files?.length) {
      onFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e) => {
    if (e.target.files?.length) {
      onFiles(e.target.files);
    }
    e.target.value = '';
  };

  const hiddenInput = (
    <input
      ref={inputRef}
      type="file"
      multiple
      accept=".pdf"
      onChange={handleChange}
      disabled={disabled}
      className="hidden-file-input"
    />
  );

  if (compact) {
    return (
      <div className="upload-compact">
        <button className="nav-item" onClick={openPicker} disabled={disabled}>
          <UploadCloudIcon />
          Add more documents
        </button>
        {hiddenInput}
      </div>
    );
  }

  return (
    <div
      className={`dropzone ${dragActive ? 'drag-active' : ''}`}
      onClick={openPicker}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
    >
      <UploadCloudIcon size={22} />
      <div className="dropzone-label">Drop PDF files here, or click to browse</div>
      <div className="dropzone-hint">PDF only &middot; multiple files supported</div>
      {hiddenInput}
    </div>
  );
}

export default UploadDropzone;
