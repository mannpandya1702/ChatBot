-- LOCAL TEST HARNESS ONLY — never deployed.
-- Recreates the slice of a hosted Supabase project that our migrations and RLS
-- tests rely on: the anon/authenticated/service_role roles, the auth schema
-- with auth.users + auth.uid(), and Supabase's default table grants.
-- On a real project all of this already exists; migrations apply unchanged.

do $$
begin
  if not exists (select from pg_roles where rolname = 'anon') then
    create role anon nologin;
  end if;
  if not exists (select from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
  if not exists (select from pg_roles where rolname = 'service_role') then
    create role service_role nologin bypassrls;  -- matches Supabase: service_role bypasses RLS
  end if;
end
$$;

grant anon, authenticated, service_role to current_user;

create schema if not exists auth;

create table if not exists auth.users (
  id         uuid primary key,
  email      text unique,
  created_at timestamptz not null default now()
);

-- Supabase resolves auth.uid() from the request JWT; locally we read the same
-- claim from a session setting, which the test harness sets per simulated user.
create or replace function auth.uid()
returns uuid
language sql stable
as $$
  select nullif(current_setting('request.jwt.claims', true)::jsonb ->> 'sub', '')::uuid
$$;

grant usage on schema auth to anon, authenticated, service_role;
grant execute on function auth.uid() to anon, authenticated, service_role;
grant select on auth.users to service_role;

-- Supabase default privileges: tables/sequences/functions created later in
-- public are granted to the three API roles (RLS then restricts rows).
grant usage on schema public to anon, authenticated, service_role;
alter default privileges in schema public
  grant all on tables to anon, authenticated, service_role;
alter default privileges in schema public
  grant all on sequences to anon, authenticated, service_role;
