import type { ReactNode } from "react";

export const metadata = {
  title: "Sainik Sahayak",
  description: "Internal knowledge assistant — answers only from the approved knowledge base.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
