import { redirect } from "next/navigation";

// Placeholder root. The middleware funnels authenticated users to /chat and
// everyone else to /login; the chat and auth UIs land in Phases 3–4. Until then
// the backend is exercised via POST /api/chat.
export default function Home() {
  redirect("/chat");
}
