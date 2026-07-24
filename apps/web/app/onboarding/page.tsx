import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";
import { Card } from "@/lib/ui/misc";
import { OnboardingWizard } from "./onboarding-wizard";

export const metadata = { title: "Set up your account — Sainik Sahayak" };

export default async function OnboardingPage() {
  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const [{ data: profile }, { data: aal }, { data: factors }] = await Promise.all([
    supabase.from("profiles").select("must_change_password, full_name").eq("id", user.id).single(),
    supabase.auth.mfa.getAuthenticatorAssuranceLevel(),
    supabase.auth.mfa.listFactors(),
  ]);

  const needsPassword = profile?.must_change_password === true;
  const hasVerifiedFactor = (factors?.all ?? []).some((f) => f.status === "verified");

  // Already fully set up (AAL2 + no forced change) — nothing to onboard.
  if (aal?.currentLevel === "aal2" && !needsPassword) redirect("/chat");

  return (
    <main className="flex min-h-dvh items-center justify-center bg-background px-4 py-10">
      <div className="w-full max-w-sm animate-fade-in">
        <div className="mb-6 text-center">
          <h1 className="text-xl font-semibold">Welcome{profile?.full_name ? `, ${profile.full_name}` : ""}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {hasVerifiedFactor ? "Confirm your authenticator to continue." : "Two quick steps to secure your account."}
          </p>
        </div>
        <Card className="p-6">
          <OnboardingWizard needsPassword={needsPassword} hasVerifiedFactor={hasVerifiedFactor} />
        </Card>
        <form action="/auth/signout" method="post" className="mt-4 text-center">
          <button type="submit" className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground">
            Not you? Sign out
          </button>
        </form>
      </div>
    </main>
  );
}
