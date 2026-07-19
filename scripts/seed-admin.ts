/**
 * Sainik Sahayak — create the first super_admin (Phase 0 utility).
 *
 * Usage:
 *   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
 *     npm run seed:admin -- --service-number SUP001 --name "Full Name" [--rank Col] [--unit HQ]
 *
 * Prints a one-time temporary password ONCE. First login forces a password
 * change + TOTP enrollment before any app access (Phase 3 middleware).
 * Server-side only: requires the service role key; never ship this to a client.
 */
import { createClient } from "@supabase/supabase-js";
import { randomBytes } from "node:crypto";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

const url = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const serviceNumber = arg("service-number");
const fullName = arg("name");

if (!url || !serviceKey || !serviceNumber || !fullName) {
  console.error(
    "Missing input. Required: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY env vars " +
      "and --service-number, --name flags.",
  );
  process.exit(1);
}

// 20 chars from a URL-safe alphabet — shown once, must be changed on first login.
const tempPassword = randomBytes(15).toString("base64url");

async function main() {
  const supabase = createClient(url!, serviceKey!, {
    auth: { autoRefreshToken: false, persistSession: false },
  });

  const email = `${serviceNumber!.toLowerCase()}@sainik.internal`;
  const { data: created, error: authErr } = await supabase.auth.admin.createUser({
    email,
    password: tempPassword,
    email_confirm: true,
  });
  if (authErr || !created.user) {
    console.error("auth user creation failed:", authErr?.message);
    process.exit(1);
  }

  const { error: profileErr } = await supabase.from("profiles").insert({
    id: created.user.id,
    service_number: serviceNumber,
    full_name: fullName,
    rank: arg("rank") ?? null,
    unit: arg("unit") ?? null,
    role: "super_admin",
    access_tier: 3,
    is_active: true,
    must_change_password: true,
  });
  if (profileErr) {
    // Roll back the half-created account rather than leaving an orphan.
    await supabase.auth.admin.deleteUser(created.user.id);
    console.error("profile creation failed (auth user rolled back):", profileErr.message);
    process.exit(1);
  }

  await supabase.rpc("log_event", {
    p_event_type: "admin_seeded",
    p_detail: { service_number: serviceNumber },
  });

  console.log("super_admin created.");
  console.log(`  service number : ${serviceNumber}`);
  console.log(`  login email    : ${email}`);
  console.log(`  temp password  : ${tempPassword}`);
  console.log("Shown once — first login forces a password change + TOTP enrollment.");
}

main();
