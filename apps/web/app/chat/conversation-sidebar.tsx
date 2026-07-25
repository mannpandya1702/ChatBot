"use client";
import Link from "next/link";
import { deleteConversationAction } from "./actions";

export interface ConversationSummary {
  id: string;
  title: string | null;
  updated_at: string;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * UTC-based on purpose: this component is server-rendered and then hydrated, so
 * a locale/timezone-dependent format would produce two different strings and a
 * hydration mismatch. Same input → same output on both sides.
 */
function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

export function ConversationSidebar({
  conversations,
  activeId,
  open,
  onClose,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <>
      {/* Scrim: mobile only — on large screens the sidebar is part of the layout. */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-foreground/40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        aria-label="Chat history"
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-border bg-card transition-transform duration-200 lg:static lg:z-auto lg:w-64 lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-3">
          <span className="text-sm font-semibold">Your chats</span>
          <div className="flex items-center gap-1">
            <Link
              href="/chat"
              onClick={onClose}
              className="rounded-md border border-input px-2.5 py-1.5 text-xs font-medium hover:bg-accent"
            >
              + New
            </Link>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close chat history"
              className="rounded-md p-1.5 text-muted-foreground hover:bg-accent lg:hidden"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto p-2">
          {conversations.length === 0 ? (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground">
              No chats yet. Ask your first question.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {conversations.map((c) => {
                const title = c.title?.trim() || "New chat";
                const active = c.id === activeId;
                return (
                  <li
                    key={c.id}
                    className={`group flex items-center gap-1 rounded-md pr-1 ${
                      active ? "bg-accent" : "hover:bg-accent/60"
                    }`}
                  >
                    <Link
                      href={`/chat?c=${c.id}`}
                      onClick={onClose}
                      aria-current={active ? "page" : undefined}
                      className="min-w-0 flex-1 px-2 py-2"
                    >
                      <span className="block truncate text-sm">{title}</span>
                      <span className="block text-[10px] text-muted-foreground">
                        {shortDate(c.updated_at)}
                      </span>
                    </Link>
                    <form
                      action={deleteConversationAction}
                      onSubmit={(e) => {
                        if (!window.confirm("Delete this chat and its messages?")) e.preventDefault();
                      }}
                    >
                      <input type="hidden" name="id" value={c.id} />
                      <input type="hidden" name="activeId" value={activeId ?? ""} />
                      {/* Always visible on touch (no hover there); hover/focus-revealed on desktop. */}
                      <button
                        type="submit"
                        aria-label={`Delete chat: ${title}`}
                        className="rounded p-1 text-muted-foreground hover:text-destructive focus-visible:opacity-100 lg:opacity-0 lg:group-hover:opacity-100"
                      >
                        <TrashIcon />
                      </button>
                    </form>
                  </li>
                );
              })}
            </ul>
          )}
        </nav>

        <p className="border-t border-border px-3 py-2 text-[10px] text-muted-foreground">
          Only you can see your chats.
        </p>
      </aside>
    </>
  );
}

function TrashIcon() {
  return (
    <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M18 6 6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
