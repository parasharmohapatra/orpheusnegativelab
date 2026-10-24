// 24×24 line icons, 1.5 stroke, matching the mock's rail.
const base = { width: 22, height: 22, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true };

export const IconConvert = () => (
  <svg {...base}>
    <rect x="3" y="5" width="18" height="14" rx="1" />
    <path d="M3 9h18M3 15h18M7 5v4M12 5v4M17 5v4M7 15v4M12 15v4M17 15v4" />
  </svg>
);
export const IconAdjust = () => (
  <svg {...base}>
    <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="8" cy="17" r="2" />
  </svg>
);
export const IconColor = () => (
  <svg {...base}>
    <circle cx="9" cy="10" r="5.5" />
    <circle cx="15" cy="10" r="5.5" />
    <circle cx="12" cy="15" r="5.5" />
  </svg>
);
export const IconCurves = () => (
  <svg {...base}>
    <path d="M4 20V4M4 20h16" />
    <path d="M5 19c5 0 4-11 9-12s4-2 5-3" />
  </svg>
);
export const IconCrop = () => (
  <svg {...base}>
    <path d="M6 2v16h16M2 6h16v16" />
  </svg>
);
export const IconPresets = () => (
  <svg {...base}>
    <rect x="4" y="4" width="7" height="7" />
    <rect x="13" y="4" width="7" height="7" />
    <rect x="4" y="13" width="7" height="7" />
    <rect x="13" y="13" width="7" height="7" />
  </svg>
);
export const IconSettings = () => (
  <svg {...base}>
    <circle cx="12" cy="12" r="3" />
    <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" />
  </svg>
);
export const IconHelp = () => (
  <svg {...base}>
    <circle cx="12" cy="12" r="9" />
    <path d="M9.5 9.5a2.5 2.5 0 015 0c0 1.7-2.5 2-2.5 4M12 17h.01" />
  </svg>
);
export const IconPipette = () => (
  <svg {...base} width={16} height={16}>
    <path d="M14 4l6 6M17 7l-9 9-4 1 1-4 9-9" />
  </svg>
);
export const IconRotate = () => (
  <svg {...base} width={16} height={16}>
    <path d="M20 12a8 8 0 11-3-6.2M20 4v5h-5" />
  </svg>
);
export const IconFlip = () => (
  <svg {...base} width={16} height={16}>
    <path d="M12 3v18M8 7L3 12l5 5V7zM16 7l5 5-5 5V7z" />
  </svg>
);
