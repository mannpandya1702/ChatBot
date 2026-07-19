-- Sainik Sahayak · migration 4: SQL functions
--   log_event            security DEFINER — sole write path into audit_logs
--   hybrid_search        security INVOKER — RLS scopes chunks to the caller's tier
--   check_rate_limit     security DEFINER — 20 msgs / 5 min / user (spec §11)
--   record_query_event   security DEFINER — service-role-only analytics insert

-- ── log_event ───────────────────────────────────────────────────────────────
-- Append-only audit writes. Table-level INSERT is revoked for clients, so this
-- definer function (owner bypasses RLS) is the only way rows get in.
create or replace function public.log_event(
  p_event_type text,
  p_detail     jsonb default '{}'::jsonb,
  p_ip         inet  default null,
  p_user_agent text  default null
) returns bigint
language plpgsql security definer
set search_path = ''
as $$
declare v_id bigint;
begin
  insert into public.audit_logs (user_id, event_type, detail, ip, user_agent)
  values (auth.uid(), p_event_type, p_detail, p_ip, p_user_agent)
  returning id into v_id;
  return v_id;
end
$$;

revoke all on function public.log_event(text, jsonb, inet, text) from public, anon;
grant execute on function public.log_event(text, jsonb, inet, text) to authenticated, service_role;

-- ── hybrid_search ───────────────────────────────────────────────────────────
-- pgvector HNSW cosine + tsvector keyword search fused with RRF (k = 60).
-- SECURITY INVOKER on purpose: the chunks/documents RLS policies filter by the
-- caller's access tier automatically, so a jawan can never retrieve above tier.
create or replace function public.hybrid_search(
  p_query_embedding vector(1024),
  p_query_text      text,
  p_match_count     int default 20
) returns table (
  chunk_id     bigint,
  document_id  uuid,
  doc_title    text,
  content      text,
  page_start   int,
  page_end     int,
  section_path text,
  score        double precision
)
-- search_path pins public + extensions (hosted Supabase installs pgvector in
-- `extensions`; locally it lives in `public`) so the <=> operator resolves.
language sql security invoker stable
set search_path = public, extensions
as $$
  with vec as (
    select c.id, row_number() over (order by c.embedding <=> p_query_embedding) as rnk
    from public.chunks c
    where c.embedding is not null
    order by c.embedding <=> p_query_embedding
    limit 40
  ),
  txt as (
    select c.id,
           row_number() over (
             order by ts_rank_cd(c.content_tsv,
               websearch_to_tsquery('simple'::regconfig, p_query_text)) desc
           ) as rnk
    from public.chunks c
    where p_query_text is not null
      and length(trim(p_query_text)) > 0
      and c.content_tsv @@ websearch_to_tsquery('simple'::regconfig, p_query_text)
    limit 40
  ),
  fused as (
    select coalesce(vec.id, txt.id) as id,
           coalesce(1.0 / (60 + vec.rnk), 0) + coalesce(1.0 / (60 + txt.rnk), 0) as score
    from vec full outer join txt on vec.id = txt.id
  )
  select c.id, c.document_id, d.title, c.content,
         c.page_start, c.page_end, c.section_path, f.score
  from fused f
  join public.chunks c on c.id = f.id
  join public.documents d on d.id = c.document_id
  order by f.score desc
  limit p_match_count
$$;

revoke all on function public.hybrid_search(vector, text, int) from public, anon;
grant execute on function public.hybrid_search(vector, text, int) to authenticated, service_role;

-- ── check_rate_limit ────────────────────────────────────────────────────────
-- True while the caller is under the limit. DEFINER so it can count messages
-- regardless of RLS timing; scoped strictly to the caller's own conversations.
create or replace function public.check_rate_limit(
  p_limit  int      default 20,
  p_window interval default interval '5 minutes'
) returns boolean
language sql security definer stable
set search_path = ''
as $$
  select count(*) < p_limit
  from public.messages m
  join public.conversations c on c.id = m.conversation_id
  where c.user_id = auth.uid()
    and m.role = 'user'
    and m.created_at > now() - p_window
$$;

revoke all on function public.check_rate_limit(int, interval) from public, anon;
grant execute on function public.check_rate_limit(int, interval) to authenticated, service_role;

-- ── record_query_event ──────────────────────────────────────────────────────
-- Single write path for query analytics; callable only by the server (service
-- role). Stores the rewritten standalone query with NO user linkage.
create or replace function public.record_query_event(
  p_query_text text,
  p_language   text,
  p_refused    boolean,
  p_top_score  real default null
) returns bigint
language plpgsql security definer
set search_path = ''
as $$
declare v_id bigint;
begin
  insert into public.query_analytics (query_text, language, refused, top_score)
  values (p_query_text, p_language, p_refused, p_top_score)
  returning id into v_id;
  return v_id;
end
$$;

revoke all on function public.record_query_event(text, text, boolean, real)
  from public, anon, authenticated;
grant execute on function public.record_query_event(text, text, boolean, real) to service_role;
