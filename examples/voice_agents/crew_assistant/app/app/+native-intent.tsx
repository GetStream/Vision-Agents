// A sign-in opened in a Custom Tab comes back to the app on its scheme (`io.getstream.expocrewassistant://
// sign-in?…`), which the SDK's sign-in reads itself. Stay on the screen it was opened from rather
// than routing to a page that doesn't exist.
export const redirectSystemPath = ({ path }: { initial: boolean; path: string | null }) =>
  path?.includes('sign-in') ? null : path;
