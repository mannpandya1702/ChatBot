-- Sainik Sahayak · migration 2: tables & indexes
-- All tables live in schema public with RLS enabled in migration 3.

-- ── profiles ────────────────────────────────────────────────────────────────
create table if not exists public.profiles (
  id                    uuid primary key references auth.users (id) on delete cascade,
  service_number        text unique not null,
  full_name             text not null,
  rank                  text,
  unit                  text,
  role                  text not null default 'jawan'
                        check (role in ('super_admin', 'admin', 'jawan')),
  access_tier           int  not null default 1 check (access_tier between 1 and 3),
  is_active             boolean not null default true,
  must_change_password  boolean not null default true,
  created_by            uuid references auth.users (id),
  last_login_at         timestamptz,
  created_at            timestamptz not null default now()
);

-- ── documents ───────────────────────────────────────────────────────────────
create table if not exists public.documents (
  id                uuid primary key default gen_random_uuid(),
  title             text not null,
  original_filename text not null,
  storage_path      text not null,
  sha256            text unique not null,
  access_tier       int  not null default 1 check (access_tier between 1 and 3),
  language          text,
  status            text not null default 'processing'
                    check (status in ('processing', 'ready', 'failed')),
  page_count        int,
  chunk_count       int,
  error             text,
  uploaded_by       uuid references auth.users (id),
  created_at        timestamptz not null default now()
);

-- ── chunks ──────────────────────────────────────────────────────────────────
create table if not exists public.chunks (
  id           bigint generated always as identity primary key,
  document_id  uuid not null references public.documents (id) on delete cascade,
  chunk_index  int  not null,
  content      text not null,
  content_tsv  tsvector generated always as (to_tsvector('simple'::regconfig, content)) stored,
  embedding    vector(1024),
  page_start   int,
  page_end     int,
  section_path text,
  token_count  int,
  -- Denormalised from documents so tier filtering never needs a join in RLS.
  access_tier  int not null default 1 check (access_tier between 1 and 3),
  unique (document_id, chunk_index)
);

create index if not exists chunks_embedding_hnsw_idx on public.chunks
  using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_content_tsv_idx on public.chunks using gin (content_tsv);
create index if not exists chunks_document_id_idx on public.chunks (document_id);

-- ── conversations & messages ────────────────────────────────────────────────
create table if not exists public.conversations (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users (id) on delete cascade,
  title      text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists conversations_user_id_idx on public.conversations (user_id);

create table if not exists public.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  role            text not null check (role in ('user', 'assistant')),
  content         text not null,
  citations       jsonb,
  model           text,
  latency_ms      int,
  refused         boolean not null default false,
  feedback        smallint check (feedback in (-1, 1)),
  created_at      timestamptz not null default now()
);
create index if not exists messages_conversation_id_idx on public.messages (conversation_id);
create index if not exists messages_created_at_idx on public.messages (created_at);

-- ── audit_logs (append-only; hardened in migration 3) ───────────────────────
create table if not exists public.audit_logs (
  id         bigint generated always as identity primary key,
  user_id    uuid references auth.users (id) on delete set null,
  event_type text not null,
  detail     jsonb not null default '{}'::jsonb,
  ip         inet,
  user_agent text,
  created_at timestamptz not null default now()
);
create index if not exists audit_logs_user_id_idx on public.audit_logs (user_id);
create index if not exists audit_logs_event_type_idx on public.audit_logs (event_type);
create index if not exists audit_logs_created_at_idx on public.audit_logs (created_at);

-- ── app_settings ────────────────────────────────────────────────────────────
create table if not exists public.app_settings (
  key   text primary key,
  value jsonb not null
);

-- ── login_attempts (server-side lockout tracking, Section 5) ────────────────
-- Written/read only by server actions using the service role. No client access.
create table if not exists public.login_attempts (
  id             bigint generated always as identity primary key,
  service_number text not null,
  ip             inet,
  success        boolean not null,
  created_at     timestamptz not null default now()
);
create index if not exists login_attempts_lookup_idx
  on public.login_attempts (service_number, created_at desc);

-- ── query_analytics (Section 10 insights without violating log minimisation) ─
-- Stores the REWRITTEN standalone query text with NO user linkage, so admins
-- get "top queries" + the unanswered-queries gap backlog while conversations
-- stay owner-only and audit_logs stay metadata-only.
create table if not exists public.query_analytics (
  id             bigint generated always as identity primary key,
  query_text     text not null,
  language       text,                -- 'hi' | 'en' | 'hinglish' (classifier output)
  refused        boolean not null default false,
  top_score      real,                -- best rerank score for threshold tuning
  created_at     timestamptz not null default now()
);
create index if not exists query_analytics_created_at_idx on public.query_analytics (created_at);
create index if not exists query_analytics_refused_idx on public.query_analytics (refused, created_at);
