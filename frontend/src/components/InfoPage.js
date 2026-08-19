import React from 'react';
import { UploadCloudIcon, SearchIcon, MessageIcon, FileTextIcon } from './Icons';

const STEPS = [
  { icon: UploadCloudIcon, title: 'You upload documents', body: "Drop in one or more PDFs. Athena reads them and breaks them into organized pieces behind the scenes." },
  { icon: SearchIcon, title: 'You ask a question', body: 'Type it like you would ask a person. No special commands or keywords needed.' },
  { icon: FileTextIcon, title: 'Athena finds the relevant parts', body: "It searches only your uploaded documents for the pieces that actually relate to your question." },
  { icon: MessageIcon, title: 'You get an answer with sources', body: "The answer is written from what it found — with a 'Source' link so you can see the exact passage it came from." },
];

const FAQS = [
  {
    q: 'Where does my data go?',
    a: "Your documents always stay on this computer. When you ask a question, the relevant excerpts (not the whole document) are sent to whichever AI provider is currently active — OpenAI's servers if you've added an API key, or nowhere at all if you're using local Ollama, since that runs entirely on your machine.",
  },
  {
    q: 'What file types can I upload?',
    a: 'PDFs only, for now.',
  },
  {
    q: "Why did it say it couldn't find an answer?",
    a: "Athena only answers from what's actually in your uploaded documents — it won't make something up just to sound helpful. If the documents don't cover it, it'll tell you rather than guess.",
  },
  {
    q: 'Do I need an OpenAI account?',
    a: "No. Without an API key, Athena automatically falls back to a local model (Ollama) running on your own machine — free and private, though generally slower and less capable than a cloud model. Add a key any time from the API key button in the sidebar.",
  },
  {
    q: "What's the difference between chats?",
    a: 'Each chat has its own separate set of documents. Uploading, asking, or deleting in one chat never affects any other. You can also create a new chat as a copy of an existing one to branch off its documents without re-uploading them.',
  },
  {
    q: 'I deleted a document or a whole chat by mistake — can I get it back?',
    a: 'No. Deleting is permanent and there is no undo, so double-check before confirming a delete.',
  },
];

function InfoPage() {
  return (
    <div className="docs-page">
      <section className="docs-hero">
        <h1>What is Athena?</h1>
        <p>
          Athena is a document Q&amp;A assistant. Upload your PDFs, then ask questions about
          them in plain English — Athena answers using only what's actually written in your
          documents, and shows you exactly where each answer came from.
        </p>
      </section>

      <section className="docs-section">
        <h2>How it works</h2>
        <div className="docs-steps">
          {STEPS.map((step, i) => (
            <div className="docs-step" key={step.title}>
              <div className="docs-step-icon"><step.icon /><span className="docs-step-num">{i + 1}</span></div>
              <div>
                <div className="docs-step-title">{step.title}</div>
                <div className="docs-step-body">{step.body}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="docs-section">
        <h2>Frequently asked questions</h2>
        <div className="docs-faq">
          {FAQS.map((item) => (
            <div className="docs-faq-item" key={item.q}>
              <div className="docs-faq-q">{item.q}</div>
              <div className="docs-faq-a">{item.a}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default InfoPage;
