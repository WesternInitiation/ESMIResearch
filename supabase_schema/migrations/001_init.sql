-- ESMI Research: Supabase schema for shared compression runs
-- Apply in the Supabase SQL editor (or via supabase db push).
--
-- After running this migration, create a private Storage bucket named `esmi-images`
-- in the Supabase dashboard (Storage → New bucket → Private).
-- The Streamlit app uses the service_role key, so bucket policies are optional for v1.

create extension if not exists "pgcrypto";

-- Primary compression run record (one save per compression experiment).
create table if not exists public.runs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    method text not null,
    source_filename text,
    archive_member text,
    params jsonb not null default '{}'::jsonb,
    original_storage_path text,
    compressed_storage_path text,
    runtime_seconds double precision,
    original_bytes bigint,
    compressed_bytes_estimate bigint,
    compression_ratio double precision,
    share_token uuid not null unique default gen_random_uuid(),
    notes text
);

create index if not exists runs_created_at_idx on public.runs (created_at desc);
create index if not exists runs_share_token_idx on public.runs (share_token);

-- Per-band quality metrics for a run.
create table if not exists public.band_metrics (
    id bigserial primary key,
    run_id uuid not null references public.runs (id) on delete cascade,
    band text not null,
    rmse double precision,
    mae double precision,
    psnr_db double precision,
    ssim double precision,
    unique (run_id, band)
);

create index if not exists band_metrics_run_id_idx on public.band_metrics (run_id);

-- Optional NDVI preservation result (at most one per run).
create table if not exists public.ndvi_results (
    run_id uuid primary key references public.runs (id) on delete cascade,
    red_band text not null,
    nir_band text not null,
    rmse double precision,
    mae double precision,
    correlation double precision,
    ssim double precision,
    bias double precision,
    created_at timestamptz not null default now()
);

-- All-method comparison snapshot (linked to a primary run when available).
create table if not exists public.method_comparisons (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    run_id uuid references public.runs (id) on delete set null,
    share_token uuid not null unique default gen_random_uuid(),
    params jsonb not null default '{}'::jsonb,
    results jsonb not null default '[]'::jsonb
);

create index if not exists method_comparisons_run_id_idx
    on public.method_comparisons (run_id);
create index if not exists method_comparisons_created_at_idx
    on public.method_comparisons (created_at desc);

-- Notes for operators:
-- 1. Create private Storage bucket: esmi-images
-- 2. Prefer Vercel (repo root Next.js) env vars:
--    NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
--    SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
--    NEXT_PUBLIC_SUPABASE_BUCKET=esmi-images
-- 3. Optional Streamlit local secrets:
--    [supabase]
--    url = "https://YOUR_PROJECT.supabase.co"
--    service_role_key = "YOUR_SERVICE_ROLE_KEY"
--    bucket = "esmi-images"
-- 4. Sharing is via ?run=<share_token> on the Vercel (or Streamlit) app.
