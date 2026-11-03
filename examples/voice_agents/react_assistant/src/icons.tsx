import type { SVGProps } from "react";

/** Line icons, 20 px on a 24 px grid, drawn in the text colour. */
function Icon({ children, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      aria-hidden
      fill="none"
      height={20}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.8}
      viewBox="0 0 24 24"
      width={20}
      {...props}
    >
      {children}
    </svg>
  );
}

export const MenuIcon = () => (
  <Icon>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </Icon>
);

export const SidebarIcon = () => (
  <Icon>
    <rect height="16" rx="3" width="18" x="3" y="4" />
    <path d="M9 4v16" />
  </Icon>
);

export const NewChatIcon = () => (
  <Icon>
    <path d="M12 4H7a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-5" />
    <path d="M17.5 3.5a2.1 2.1 0 0 1 3 3L13 14l-4 1 1-4z" />
  </Icon>
);

export const TrashIcon = () => (
  <Icon height={16} width={16}>
    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" />
  </Icon>
);

export const SunIcon = () => (
  <Icon>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Icon>
);

export const MoonIcon = () => (
  <Icon>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
  </Icon>
);

export const CloseIcon = () => (
  <Icon>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);

export const MicIcon = () => (
  <Icon>
    <rect height="12" rx="3" width="6" x="9" y="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </Icon>
);

export const MicOffIcon = () => (
  <Icon>
    <path d="M15 9.3V6a3 3 0 0 0-5.7-1.3M9 9v3a3 3 0 0 0 4.9 2.3M5 11a7 7 0 0 0 11.4 5.4M19 11a7 7 0 0 1-.6 2.9M12 18v3M3 3l18 18" />
  </Icon>
);

export const CameraIcon = () => (
  <Icon>
    <rect height="12" rx="2.5" width="13" x="2" y="6" />
    <path d="M15 10.5l6-3.5v10l-6-3.5" />
  </Icon>
);

export const CameraOffIcon = () => (
  <Icon>
    <path d="M10 6h2.5A2.5 2.5 0 0 1 15 8.5v2l6-3.5v10l-3.6-2.1M15 15.5A2.5 2.5 0 0 1 12.5 18h-8A2.5 2.5 0 0 1 2 15.5v-7A2.5 2.5 0 0 1 4.5 6M3 3l18 18" />
  </Icon>
);

export const HangUpIcon = () => (
  <Icon fill="currentColor" height={22} strokeWidth={0} width={22}>
    <path d="M3.5 14.5c-.6-.6-.6-1.6 0-2.2C5.9 9.9 8.9 8.7 12 8.7s6.1 1.2 8.5 3.6c.6.6.6 1.6 0 2.2l-1.6 1.6a1.2 1.2 0 0 1-1.6.1l-2-1.6a1.2 1.2 0 0 1-.4-1v-1.8a10 10 0 0 0-5.8 0v1.8c0 .4-.2.8-.5 1l-2 1.6a1.2 1.2 0 0 1-1.6-.1z" />
  </Icon>
);
