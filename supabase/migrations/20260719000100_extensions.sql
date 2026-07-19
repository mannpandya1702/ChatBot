-- Sainik Sahayak · migration 1: extensions & private helper schema
-- Applies identically on hosted Supabase (ap-south-1) and local Postgres 16+pgvector.

create extension if not exists vector;

-- Internal helper schema: security-definer lookups used by RLS policies.
-- Not exposed through PostgREST.
create schema if not exists private;
