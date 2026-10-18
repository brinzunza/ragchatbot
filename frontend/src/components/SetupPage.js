import React from 'react';
import { KeyIcon, PlusIcon, UploadCloudIcon, MessageIcon, FileTextIcon } from './Icons';

const WALKTHROUGH = [
  { icon: PlusIcon, title: 'create a chat', body: 'Click "+ New chat" in the sidebar and give it a name. Choose "Start empty," or "Copy from" to branch off an existing chat\'s documents.' },
  { icon: UploadCloudIcon, title: 'upload a pdf', body: 'Drop a file onto the upload area, or click it to browse. A progress bar tracks the conversion — you can keep chatting in other chats while it works.' },
  { icon: MessageIcon, title: 'ask a question', body: 'Type it into the box at the bottom, like you\'re asking a person. Athena answers using only what it found in that chat\'s documents.' },
  { icon: FileTextIcon, title: 'check the source', body: 'Click "Source" under an answer to open the original document with the exact passage highlighted.' },
];

function SetupPage({ apiKeyStatus, onOpenApiKey, onNewChat }) {
  return (
    <div className="docs-page">
      <section className="docs-hero">
        <h1>setup guide</h1>
        <p>get your AI provider connected, then create your first chat.</p>
      </section>

      <section className="docs-section">
        <h2>1. choose your AI provider</h2>

        {apiKeyStatus && (
          <div className="api-key-status docs-status">
            <span className={`status-dot ${apiKeyStatus.has_key ? 'active' : ''}`} />
            currently using {apiKeyStatus.provider} ({apiKeyStatus.model})
          </div>
        )}

        <div className="docs-cards">
          <div className="docs-card">
            <div className="docs-card-title">option A — OpenAI (cloud)</div>
            <ol className="docs-card-list">
              <li>create a key at <code>platform.openai.com/api-keys</code></li>
              <li>open the provider settings below and paste it under OpenAI</li>
            </ol>
            <p className="docs-card-note">faster and generally higher quality. requires an OpenAI account and usage costs money.</p>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">option B — Claude (cloud)</div>
            <ol className="docs-card-list">
              <li>create a key at <code>console.anthropic.com</code></li>
              <li>open the provider settings below and paste it under Claude</li>
            </ol>
            <p className="docs-card-note">strong at following instructions and using tools. requires an Anthropic account with credits; usage costs money.</p>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">option C — Ollama (local)</div>
            <ol className="docs-card-list">
              <li>install Ollama from <code>ollama.com</code></li>
              <li>run <code>ollama pull llama3.2</code> in a terminal</li>
              <li>choose Ollama in the provider settings (or leave the API keys empty — Athena detects this automatically)</li>
            </ol>
            <p className="docs-card-note">free and fully private — nothing leaves your machine. slower and less capable than a cloud model.</p>
          </div>
        </div>

        <button className="nav-item primary-action docs-cta" onClick={onOpenApiKey}>
          <KeyIcon />
          open provider settings
        </button>
      </section>

      <section className="docs-section">
        <h2>2. create your first chat</h2>
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
          start a new chat
        </button>
      </section>
    </div>
  );
}

export default SetupPage;
