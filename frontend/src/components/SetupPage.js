import React from 'react';
import { KeyIcon, PlusIcon, UploadCloudIcon, MessageIcon, FileTextIcon } from './Icons';

const WALKTHROUGH = [
  { icon: PlusIcon, title: 'Create a chat', body: 'Click "+ New chat" in the sidebar and give it a name. Choose "Start empty," or "Copy from" to branch off an existing chat\'s documents.' },
  { icon: UploadCloudIcon, title: 'Upload a PDF', body: 'Drop a file onto the upload area, or click it to browse. A progress bar tracks the conversion — you can keep chatting in other chats while it works.' },
  { icon: MessageIcon, title: 'Ask a question', body: 'Type it into the box at the bottom, like you\'re asking a person. Athena answers using only what it found in that chat\'s documents.' },
  { icon: FileTextIcon, title: 'Check the source', body: 'Click "Source" under an answer to open the original document with the exact passage highlighted.' },
];

function SetupPage({ apiKeyStatus, onOpenApiKey, onNewChat }) {
  return (
    <div className="docs-page">
      <section className="docs-hero">
        <h1>Setup guide</h1>
        <p>Get your AI provider connected, then create your first chat.</p>
      </section>

      <section className="docs-section">
        <h2>1. Choose your AI provider</h2>

        {apiKeyStatus && (
          <div className="api-key-status docs-status">
            <span className={`status-dot ${apiKeyStatus.has_key ? 'active' : ''}`} />
            Currently using {apiKeyStatus.provider} ({apiKeyStatus.model})
          </div>
        )}

        <div className="docs-cards">
          <div className="docs-card">
            <div className="docs-card-title">Option A — OpenAI (cloud)</div>
            <ol className="docs-card-list">
              <li>Create a key at <code>platform.openai.com/api-keys</code></li>
              <li>Open the API key settings below and paste it in</li>
            </ol>
            <p className="docs-card-note">Faster and generally higher quality. Requires an OpenAI account and usage costs money.</p>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Option B — Ollama (local)</div>
            <ol className="docs-card-list">
              <li>Install Ollama from <code>ollama.com</code></li>
              <li>Run <code>ollama pull llama3.2</code> in a terminal</li>
              <li>Leave the API key empty — Athena detects this automatically</li>
            </ol>
            <p className="docs-card-note">Free and fully private — nothing leaves your machine. Slower and less capable than a cloud model.</p>
          </div>
        </div>

        <button className="nav-item primary-action docs-cta" onClick={onOpenApiKey}>
          <KeyIcon />
          Open API key settings
        </button>
      </section>

      <section className="docs-section">
        <h2>2. Create your first chat</h2>
        <div className="docs-steps">
          {WALKTHROUGH.map((step, i) => (
            <div className="docs-step" key={step.title}>
              <div className="docs-step-icon"><step.icon /><span className="docs-step-num">{i + 1}</span></div>
              <div>
                <div className="docs-step-title">{step.title}</div>
                <div className="docs-step-body">{step.body}</div>
              </div>
            </div>
          ))}
        </div>

        <button className="nav-item primary-action docs-cta" onClick={onNewChat}>
          <PlusIcon />
          Start a new chat
        </button>
      </section>
    </div>
  );
}

export default SetupPage;
