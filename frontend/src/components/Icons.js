import React from 'react';

const base = {
  width: 15,
  height: 15,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

export const TrashIcon = () => (
  <svg {...base}>
    <path d="M3 4.5h10M6 4.5V3a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1.5M6.5 7.5v4M9.5 7.5v4M4 4.5l.6 8.1a1 1 0 0 0 1 .9h4.8a1 1 0 0 0 1-.9l.6-8.1" />
  </svg>
);

export const RefreshIcon = () => (
  <svg {...base}>
    <path d="M13 8a5 5 0 1 1-1.6-3.66M13 2.5v3h-3" />
  </svg>
);

export const UploadCloudIcon = ({ size = 15 }) => (
  <svg {...base} width={size} height={size}>
    <path d="M4.8 11.5A2.8 2.8 0 0 1 5.2 6a3.6 3.6 0 0 1 6.9-1.2A2.7 2.7 0 0 1 11.5 10" style={{ fill: 'none' }} />
    <path d="M8 6.5v6M5.7 9.9 8 7.6l2.3 2.3" />
  </svg>
);

export const ChevronRightIcon = () => (
  <svg {...base} width={10} height={10}>
    <path d="M5 3l4 5-4 5" />
  </svg>
);

export const SendIcon = () => (
  <svg {...base} width={14} height={14} stroke="none" fill="currentColor">
    <path d="M1.5 8 13.5 2.5 9.5 8l4 5.5L1.5 8Z" />
  </svg>
);

export const FileTextIcon = () => (
  <svg {...base}>
    <path d="M4.5 1.5h4.8L12 4.2v9.3a1 1 0 0 1-1 1h-6a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1Z" />
    <path d="M9 1.5v3h3M6 8h4M6 10.5h4" />
  </svg>
);

export const CloseIcon = () => (
  <svg {...base} width={14} height={14}>
    <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
  </svg>
);

export const SearchIcon = () => (
  <svg {...base} width={14} height={14}>
    <circle cx="6.8" cy="6.8" r="4.3" />
    <path d="M13 13l-3-3" />
  </svg>
);
