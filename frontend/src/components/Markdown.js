import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Links open in a new tab, and wide tables scroll sideways instead of
// stretching the page. Raw HTML in the text is never rendered.
const components = {
  // eslint-disable-next-line jsx-a11y/anchor-has-content -- the content arrives as children via props
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  table: ({ node, ...props }) => (
    <div className="md-table">
      <table {...props} />
    </div>
  ),
};

function Markdown({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}

export default Markdown;
