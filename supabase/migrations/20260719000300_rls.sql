-- Sainik Sahayak · migration 3: Row-Level Security — the real security boundary.
-- Deny by default: RLS enabled on every table; only the policies below open access.
-- Server-side writes (ingestion, chat pipeline, invites) use the service role,
-- which carries BYPASSRLS — hence several tables define NO write policies at all.

-- ── security-definer helpers (bypass RLS for profile lookups inside policies) ─
create or replace function private.user_role()
returns text language sql security definer stable
set search_path = ''
as $$
  select p.role from public.profiles p where p.id = auth.uid()
$$;

create or replace function private.user_tier()
returns int language sql security definer stable
set search_path = ''
as $$
  select p.access_tier from public.profiles p where p.id = auth.uid()
$$;

create or replace function private.user_is_active()
returns boolean language sql security definer stable
set search_path = ''
as $$
  select coalesce(
    (select p.is_active from public.profiles p where p.id = auth.uid()),
    false
  )
$$;

create or replace function private.doc_is_ready(p_document_id uuid)
returns boolean language sql security definer stable
set search_path = ''
as $$
  select exists (
    select 1 from public.documents d
    where d.id = p_document_id and d.status = 'ready'
  )
$$;

revoke all on function private.user_role(),
              private.user_tier(),
              private.user_is_active(),
              private.doc_is_ready(uuid) from public;
grant execute on function private.user_role(),
                          private.user_tier(),
                          private.user_is_active(),
                          private.doc_is_ready(uuid) to authenticated;
grant usage on schema private to authenticated;

-- ── enable RLS everywhere ────────────────────────────────────────────────────
alter table public.profiles        enable row level security;
alter table public.documents       enable row level security;
alter table public.chunks          enable row level security;
alter table public.conversations   enable row level security;
alter table public.messages        enable row level security;
alter table public.audit_logs      enable row level security;
alter table public.app_settings    enable row level security;
alter table public.login_attempts  enable row level security;  -- no policies: service role only
alter table public.query_analytics enable row level security;

-- ── profiles ────────────────────────────────────────────────────────────────
-- Read: own row, or any row for admin/super_admin. Deactivated users read nothing.
drop policy if exists profiles_select_own on public.profiles;
create policy profiles_select_own on public.profiles
  for select to authenticated
  using (id = auth.uid() and private.user_is_active());

drop policy if exists profiles_select_admin on public.profiles;
create policy profiles_select_admin on public.profiles
  for select to authenticated
  using (private.user_role() in ('admin', 'super_admin') and private.user_is_active());

-- Update: super_admin any row; admin only jawan rows (column guard in trigger below).
-- No self-update, no INSERT, no DELETE via client JWTs (invite/reset = service role).
drop policy if exists profiles_update_super_admin on public.profiles;
create policy profiles_update_super_admin on public.profiles
  for update to authenticated
  using (private.user_role() = 'super_admin' and private.user_is_active())
  with check (private.user_role() = 'super_admin');

drop policy if exists profiles_update_admin_on_jawans on public.profiles;
create policy profiles_update_admin_on_jawans on public.profiles
  for update to authenticated
  using (private.user_role() = 'admin' and private.user_is_active() and role = 'jawan')
  with check (role = 'jawan');

-- Column guard: via client JWTs, only super_admin may change role/access_tier;
-- admins may only flip is_active. id/service_number/created_at are immutable
-- for every JWT caller. Service-role paths (auth.uid() is null) are exempt.
create or replace function private.guard_profile_update()
returns trigger language plpgsql security definer
set search_path = ''
as $$
declare v_caller_role text;
begin
  if auth.uid() is null then
    return new;  -- service role / migrations
  end if;
  v_caller_role := private.user_role();
  if new.id is distinct from old.id
     or new.service_number is distinct from old.service_number
     or new.created_at is distinct from old.created_at
     or new.created_by is distinct from old.created_by then
    raise exception 'immutable profile fields cannot be changed';
  end if;
  if v_caller_role is distinct from 'super_admin'
     and (new.role is distinct from old.role
          or new.access_tier is distinct from old.access_tier) then
    raise exception 'only super_admin may change role or access_tier';
  end if;
  if v_caller_role = 'admin'
     and (new.full_name is distinct from old.full_name
          or new.rank is distinct from old.rank
          or new.unit is distinct from old.unit
          or new.must_change_password is distinct from old.must_change_password
          or new.last_login_at is distinct from old.last_login_at) then
    raise exception 'admin may only change is_active on jawan profiles';
  end if;
  return new;
end
$$;

drop trigger if exists profiles_guard_update on public.profiles;
create trigger profiles_guard_update
  before update on public.profiles
  for each row execute function private.guard_profile_update();

-- ── documents & chunks ──────────────────────────────────────────────────────
-- SELECT only where status = 'ready' and tier fits the caller (spec §4 matrix).
-- Admin-console listings (incl. processing/failed) read via service role
-- server-side; no client-JWT write path exists for either table.
drop policy if exists documents_select_ready_tier on public.documents;
create policy documents_select_ready_tier on public.documents
  for select to authenticated
  using (
    status = 'ready'
    and access_tier <= private.user_tier()
    and private.user_is_active()
  );

drop policy if exists chunks_select_ready_tier on public.chunks;
create policy chunks_select_ready_tier on public.chunks
  for select to authenticated
  using (
    access_tier <= private.user_tier()
    and private.doc_is_ready(document_id)
    and private.user_is_active()
  );

-- ── conversations ───────────────────────────────────────────────────────────
-- Owner-only for every operation. Admins do NOT read chat content (spec §4).
drop policy if exists conversations_owner_all on public.conversations;
create policy conversations_owner_all on public.conversations
  for all to authenticated
  using (user_id = auth.uid() and private.user_is_active())
  with check (user_id = auth.uid() and private.user_is_active());

-- ── messages ────────────────────────────────────────────────────────────────
-- Owner reads via conversation ownership. INSERT/DELETE happen only in the
-- server-side chat pipeline / retention job (service role): a client JWT can
-- never write chat content, which keeps the citation post-check trustworthy.
-- Owner UPDATE is allowed solely so thumbs feedback works; a trigger pins it
-- to the feedback column.
drop policy if exists messages_select_owner on public.messages;
create policy messages_select_owner on public.messages
  for select to authenticated
  using (
    private.user_is_active()
    and exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

drop policy if exists messages_update_owner_feedback on public.messages;
create policy messages_update_owner_feedback on public.messages
  for update to authenticated
  using (
    private.user_is_active()
    and exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

create or replace function private.guard_message_update()
returns trigger language plpgsql security definer
set search_path = ''
as $$
begin
  if auth.uid() is null then
    return new;  -- service role
  end if;
  if new.id              is distinct from old.id
     or new.conversation_id is distinct from old.conversation_id
     or new.role         is distinct from old.role
     or new.content      is distinct from old.content
     or new.citations::text is distinct from old.citations::text
     or new.model        is distinct from old.model
     or new.latency_ms   is distinct from old.latency_ms
     or new.refused      is distinct from old.refused
     or new.created_at   is distinct from old.created_at then
    raise exception 'clients may only change message feedback';
  end if;
  return new;
end
$$;

drop trigger if exists messages_guard_update on public.messages;
create trigger messages_guard_update
  before update on public.messages
  for each row execute function private.guard_message_update();

-- ── audit_logs: append-only at the DB level ─────────────────────────────────
-- Reads: admin & super_admin. Writes: ONLY via log_event() (migration 4).
drop policy if exists audit_logs_select_admin on public.audit_logs;
create policy audit_logs_select_admin on public.audit_logs
  for select to authenticated
  using (private.user_role() in ('admin', 'super_admin') and private.user_is_active());

revoke insert, update, delete on public.audit_logs from authenticated, anon;

-- ── app_settings ────────────────────────────────────────────────────────────
drop policy if exists app_settings_select_authenticated on public.app_settings;
create policy app_settings_select_authenticated on public.app_settings
  for select to authenticated
  using (private.user_is_active());

drop policy if exists app_settings_write_super_admin on public.app_settings;
create policy app_settings_write_super_admin on public.app_settings
  for all to authenticated
  using (private.user_role() = 'super_admin' and private.user_is_active())
  with check (private.user_role() = 'super_admin');

-- ── query_analytics ─────────────────────────────────────────────────────────
-- Inserted server-side only (service role / record_query_event). Admin+ read.
drop policy if exists query_analytics_select_admin on public.query_analytics;
create policy query_analytics_select_admin on public.query_analytics
  for select to authenticated
  using (private.user_role() in ('admin', 'super_admin') and private.user_is_active());

revoke insert, update, delete on public.query_analytics from authenticated, anon;

-- login_attempts: RLS enabled with zero policies + zero grants = service role only.
revoke all on public.login_attempts from authenticated, anon;

-- Anon gets nothing anywhere: no policy above targets anon, and PostgREST table
-- grants are stripped for belt-and-braces:
revoke all on public.profiles, public.documents, public.chunks,
           public.conversations, public.messages, public.app_settings,
           public.query_analytics, public.audit_logs from anon;
