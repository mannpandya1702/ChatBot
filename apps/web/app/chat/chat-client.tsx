"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Button } from "@/lib/ui/button";
import { Spinner } from "@/lib/ui/misc";

interface Citation {
  s: number;
  document_title: string;
  page_start: number | null;
  page_end: number | null;
}
interface Msg {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  refused?: boolean;
}
interface User {
  fullName: string;
  serviceNumber: string;
  role: string;
}

const MAX = 2000;
const EXAMPLES = [
  "How do I apply for annual leave?",
  "पारिवारिक पेंशन का दावा कैसे करें?",
  "What documents are needed for an ECHS card?",
];

function pages(c: Citation): string {
  if (c.page_start == null) return "";
  if (c.page_end == null || c.page_end === c.page_start) return ` · p. ${c.page_start}`;
  return ` · pp. ${c.page_start}–${c.page_end}`;
}

export function ChatClient({ user }: { user: User }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const isAdmin = user.role === "admin" || user.role === "super_admin";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  // Grow the composer to fit its content (capped ~6 lines), shrink back on reset.
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || pending) return;
    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: message }]);
    setPending(true);
    let failed = false;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, conversationId: conversationId ?? undefined }),
      });
      if (res.status === 401) {
        window.location.href = "/login"; // session expired — send them to sign in
        return;
      }
      if (!res.ok) {
        failed = true;
        setError(
          res.status === 429
            ? "You’re sending messages too quickly. Please wait a moment. / कृपया थोड़ी देर प्रतीक्षा करें।"
            : "Something went wrong. Please try again. / कुछ गड़बड़ हुई, पुनः प्रयास करें।",
        );
        return;
      }
      const data = await res.json();
      setConversationId(data.conversationId ?? conversationId);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.text, citations: data.citations, refused: data.refused },
      ]);
    } catch {
      failed = true;
      setError("Network error. Please check your connection. / नेटवर्क त्रुटि।");
    } finally {
      setPending(false);
      if (failed) {
        // drop the unanswered bubble and hand the user their text back to retry
        setMessages((m) => (m[m.length - 1]?.role === "user" ? m.slice(0, -1) : m));
        setInput(message);
      }
      taRef.current?.focus();
    }
  }

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border bg-card px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-sm font-bold text-primary-foreground" aria-hidden="true">सै</span>
          <div className="leading-tight">
            <div className="text-sm font-semibold">Sainik Sahayak</div>
            <div className="text-xs text-muted-foreground">{user.fullName}{user.serviceNumber ? ` · ${user.serviceNumber}` : ""}</div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {isAdmin && (
            <Link href="/admin" className="rounded-md px-3 py-1.5 text-sm font-medium text-foreground hover:bg-accent">Admin</Link>
          )}
          <form action="/auth/signout" method="post">
            <Button type="submit" variant="ghost" size="sm">Sign out</Button>
          </form>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col overflow-y-auto px-4 py-4">
        {/* Always-mounted live region so screen readers announce every turn. */}
        <div className="sr-only" role="status" aria-live="polite">
          {pending ? "Searching the knowledge base" : lastAssistant?.content ?? ""}
        </div>
        {messages.length === 0 ? (
          <div className="m-auto max-w-md text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-accent text-2xl" aria-hidden="true">🎖️</div>
            <h1 className="text-lg font-semibold">Ask about welfare, leave, pension & SOPs</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              कल्याण, अवकाश, पेंशन के बारे में पूछें। Answers come only from approved documents, with sources.
            </p>
            <div className="mt-5 space-y-2">
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="block w-full rounded-md border border-border bg-card px-3 py-2 text-left text-sm hover:bg-accent"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((m, i) => (
              <MessageBubble key={i} m={m} />
            ))}
            {pending && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="text-primary" /> Searching the knowledge base…
              </div>
            )}
            <div ref={endRef} />
          </div>
        )}
      </main>

      <footer className="border-t border-border bg-card px-4 py-3">
        <div className="mx-auto w-full max-w-2xl">
          {error && <div role="alert" className="mb-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</div>}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="flex items-end gap-2"
          >
            <textarea
              ref={taRef}
              value={input}
              onChange={(e) => setInput(e.target.value.slice(0, MAX))}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              rows={1}
              placeholder="Type your question… / अपना प्रश्न लिखें…"
              aria-label="Type your question / अपना प्रश्न लिखें"
              className="max-h-40 min-h-[2.5rem] flex-1 resize-none rounded-md border border-input bg-card px-3 py-2 text-base sm:text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            <Button type="submit" size="icon" disabled={pending || !input.trim()} aria-label="Send">
              {pending ? <Spinner /> : <SendIcon />}
            </Button>
          </form>
          <div className="mt-1 text-right text-[10px] text-muted-foreground">{input.length}/{MAX}</div>
        </div>
      </footer>
    </div>
  );
}

function MessageBubble({ m }: { m: Msg }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-primary px-4 py-2 text-sm text-primary-foreground">{m.content}</div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div className={`max-w-[90%] rounded-2xl rounded-bl-sm border px-4 py-3 text-sm ${m.refused ? "border-amber-500/40 bg-amber-500/10" : "border-border bg-card"}`}>
        <div className="whitespace-pre-wrap leading-relaxed text-foreground">{m.content}</div>
        {m.citations && m.citations.length > 0 && (
          <div className="mt-3 border-t border-border pt-2">
            <div className="mb-1 text-xs font-semibold text-muted-foreground">Sources / स्रोत</div>
            <ul className="space-y-0.5">
              {m.citations.map((c) => (
                <li key={c.s} className="text-xs text-muted-foreground">
                  <span className="font-mono text-primary">[S{c.s}]</span> {c.document_title}{pages(c)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
