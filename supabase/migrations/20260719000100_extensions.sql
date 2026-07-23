-- Sainik Sahayak · migration 1: extensions & private helper schema
-- Applies identically on hosted Supabase (ap-south-1) and local Postgres 16+pgvector.

-- Pin pgvector to `public` explicitly. On a fresh Supabase project an
-- unqualified `create extension vector` can land in the `extensions` schema,
-- which would leave the unqualified `vector(1024)` / `vector_cosine_ops`
-- references below unresolved. Forcing `public` matches the tested local setup
-- and keeps `supabase db push` behaviour identical. (Assumes pgvector is not
-- already enabled in another schema — true for a new project.)
create extension if not exists vector with schema public;

-- Internal helper schema: security-definer lookups used by RLS policies.
-- Not exposed through PostgREST.
create schema if not exists private;
