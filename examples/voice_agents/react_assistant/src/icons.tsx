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
