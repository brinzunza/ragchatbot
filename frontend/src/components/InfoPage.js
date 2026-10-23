import React from 'react';
import { UploadCloudIcon, SearchIcon, MessageIcon, FileTextIcon } from './Icons';

const STEPS = [
  { icon: UploadCloudIcon, title: 'you upload documents', body: "Drop in one or more PDFs. Athena reads them and breaks them into organized pieces behind the scenes." },
  { icon: SearchIcon, title: 'you ask a question', body: 'Type it like you would ask a person. No special commands or keywords needed.' },
  { icon: FileTextIcon, title: 'athena finds the relevant parts', body: "It searches only your uploaded documents for the pieces that actually relate to your question." },
  { icon: MessageIcon, title: 'you get an answer with sources', body: "The answer is written from what it found — with a 'Source' link so you can see the exact passage it came from." },
];

const FAQS = [
  {
    q: 'where does my data go?',
    a: "Your documents always stay on this computer. When you ask a question, the relevant excerpts (not the whole document) are sent to whichever AI provider is currently active — OpenAI's or Anthropic's servers if you've added a key for one of them, or nowhere at all if you're using local Ollama, since that runs entirely on your machine. If Athena searches the web, the search words it chose go to a public search engine; you can switch web search off in the agent settings.",
  },
  {
    q: 'what file types can I upload?',
    a: 'PDFs only, for now.',
  },
  {
    q: "why did it say it couldn't find an answer?",
    a: "When a question is about your documents, Athena searches them and shows the passages it used. If they don't cover it, it will say what it couldn't find rather than guess — or look on the web if web search is on.",
  },
  {
    q: 'do I need an OpenAI or Claude account?',
    a: "No. Without an API key, Athena automatically falls back to a local model (Ollama) running on your own machine — free and private, though generally slower and less capable than a cloud model. Add an OpenAI or Claude key any time from the ai provider button in the sidebar, and switch between providers there without losing your keys.",
  },
  {
    q: 'what can the agent do?',
    a: "Athena decides for each message which tools it needs: searching and reading your documents, searching the web and reading pages, doing exact math, checking the date, and running Python for data analysis and coding. Each tool call shows up as a step above the answer that you can expand. It uses small talk and simple questions without any tools. Open the agent settings in the sidebar to turn it off (Athena then uses the classic search-then-answer flow) or to change how many tool calls it may make per message. Small local models are less reliable at picking tools than OpenAI or Claude models.",
  },
  {
    q: 'is it safe to let it run code?',
    a: "That's what the permission mode in the agent settings is for. Safe: Athena never runs code. Ask first (the default): it shows you the exact code and waits for you to allow it each time. Bypass: code runs without asking. Code runs in a separate process inside the chat's workspace folder, with a time limit, but it is not a sandbox — it can read files and use the network like any program you run. Only use bypass with a model and documents you trust.",
  },
  {
    q: 'can I redo or edit my last message?',
    a: "Yes. Under the last reply, regenerate asks the same question again and copy puts the answer on your clipboard. Under your last message, edit puts the question back in the box. Either way the old answer is also forgotten from the context so it isn't counted twice. Replies are formatted (lists, tables, code), and a finished research report can be downloaded as a Markdown file.",
  },
  {
    q: 'what is /research?',
    a: "Type /research followed by a topic for a longer investigation. Athena breaks the topic into sub-questions, researches each with your documents and the web in the background, then writes a report. You can keep chatting, leave and come back, or cancel it. Background research only runs code if the chat is set to bypass.",
  },
  {
    q: 'what is context, and can I edit it?',
    a: "Context is what Athena remembers of your conversation so it can follow up on earlier questions. Open it from the sidebar (or type /context) to remove exchanges you don't want it to remember, or change how many recent exchanges it uses. Type /compact to boil it down to a short summary (add a focus, like /compact keep the budget details), /forget <topic> to remove everything about a topic (like /forget john), /context keep only info on <topic> to drop everything unrelated to a topic, or /context clear to start fresh. Each message sends the most recent exchanges that fit a token budget (about four characters per token); the sidebar shows how full that is, and exchanges that no longer fit can be kept as a summary, by hand or automatically.",
  },
  {
    q: "what's the difference between chats?",
    a: 'Each chat has its own separate set of documents. Uploading, asking, or deleting in one chat never affects any other. You can also create a new chat as a copy of an existing one to branch off its documents without re-uploading them.',
  },
  {
    q: 'i deleted a document or a whole chat by mistake — can I get it back?',
    a: 'No. Deleting is permanent and there is no undo, so double-check before confirming a delete.',
  },
];

function InfoPage() {
  return (
    <div className="docs-page">
      <section className="docs-hero">
        <h1>what is athena?</h1>
        <p>
          Athena is a document Q&amp;A assistant. Upload your PDFs, then ask questions about
          them in plain English — Athena answers using only what's actually written in your
          documents, and shows you exactly where each answer came from.
        </p>
      </section>

      <section className="docs-section">
        <h2>how it works</h2>
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
        <h2>frequently asked questions</h2>
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
