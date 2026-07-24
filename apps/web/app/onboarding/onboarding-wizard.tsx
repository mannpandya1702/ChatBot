"use client";
import { useActionState, useEffect, useState, useTransition } from "react";
import { Button } from "@/lib/ui/button";
import { Input } from "@/lib/ui/input";
import { Label, Alert, Spinner } from "@/lib/ui/misc";
import {
  changePasswordAction,
  enrollAction,
  verifyEnrollAction,
  challengeExistingAction,
  type EnrollState,
} from "./actions";

function Qr({ code }: { code: string }) {
  const isDataUrl = code.startsWith("data:") || code.startsWith("http");
  return (
    <div className="mx-auto flex h-44 w-44 items-center justify-center rounded-md border border-border bg-white p-2">
      {isDataUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={code} alt="Authenticator QR code" className="h-full w-full" />
      ) : (
        <div className="h-full w-full [&>svg]:h-full [&>svg]:w-full" dangerouslySetInnerHTML={{ __html: code }} />
      )}
    </div>
  );
}

function StepDots({ active, total }: { active: number; total: number }) {
  return (
    <div className="mb-5 flex justify-center gap-2">
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={i === active ? "h-2 w-6 rounded-full bg-primary" : "h-2 w-2 rounded-full bg-muted"} />
      ))}
    </div>
  );
}

function CodeField({ label }: { label: string }) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor="code">{label}</Label>
      <Input id="code" name="code" inputMode="numeric" autoComplete="one-time-code" maxLength={6} placeholder="123456" autoFocus required />
    </div>
  );
}

export function OnboardingWizard({
  needsPassword,
  hasVerifiedFactor,
}: {
  needsPassword: boolean;
  hasVerifiedFactor: boolean;
}) {
  const total = needsPassword ? 2 : 1;
  const [step, setStep] = useState<"password" | "mfa">(needsPassword ? "password" : "mfa");
  const [pw, doPw, pwPending] = useActionState(changePasswordAction, {});
  useEffect(() => {
    if (pw.ok) setStep("mfa");
  }, [pw.ok]);

  // First-time enrollment only runs when there is no existing verified factor.
  const [enroll, setEnroll] = useState<EnrollState | null>(null);
  const [, startEnroll] = useTransition();
  useEffect(() => {
    if (step === "mfa" && !hasVerifiedFactor && !enroll) {
      startEnroll(async () => {
        try {
          setEnroll(await enrollAction());
        } catch {
          setEnroll({ error: "Could not start authenticator setup. Please retry." });
        }
      });
    }
  }, [step, hasVerifiedFactor, enroll]);

  const [verify, doVerify, verifying] = useActionState(verifyEnrollAction, {});
  const [challenge, doChallenge, challenging] = useActionState(challengeExistingAction, {});

  if (step === "password") {
    return (
      <>
        <StepDots active={0} total={total} />
        <h2 className="mb-1 text-lg font-semibold">Set a new password</h2>
        <p className="mb-4 text-sm text-muted-foreground">नया पासवर्ड चुनें · at least 10 characters.</p>
        <form action={doPw} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="password">New password / नया पासवर्ड</Label>
            <Input id="password" name="password" type="password" autoComplete="new-password" autoFocus required />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="confirm">Confirm password / पुष्टि करें</Label>
            <Input id="confirm" name="confirm" type="password" autoComplete="new-password" required />
          </div>
          {pw.error && <Alert variant="error">{pw.error}</Alert>}
          <Button type="submit" className="w-full" disabled={pwPending}>
            {pwPending && <Spinner />} Continue / जारी रखें
          </Button>
        </form>
      </>
    );
  }

  // MFA step — challenge an existing authenticator, or enroll a first one.
  if (hasVerifiedFactor) {
    return (
      <>
        <StepDots active={needsPassword ? 1 : 0} total={total} />
        <h2 className="mb-1 text-lg font-semibold">Confirm your authenticator</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Enter the current 6-digit code from your authenticator app to continue.
        </p>
        <form action={doChallenge} className="space-y-4">
          <CodeField label="6-digit code / कोड" />
          {challenge.error && <Alert variant="error">{challenge.error}</Alert>}
          <Button type="submit" className="w-full" disabled={challenging}>
            {challenging && <Spinner />} Verify / सत्यापित करें
          </Button>
        </form>
      </>
    );
  }

  return (
    <>
      <StepDots active={needsPassword ? 1 : 0} total={total} />
      <h2 className="mb-1 text-lg font-semibold">Set up your authenticator</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        Scan this with Google Authenticator (or any TOTP app), then enter the 6-digit code.
      </p>
      {!enroll ? (
        <div className="flex h-44 items-center justify-center"><Spinner className="h-6 w-6 text-primary" /></div>
      ) : enroll.error ? (
        <div className="space-y-3">
          <Alert variant="error">{enroll.error}</Alert>
          <Button variant="outline" className="w-full" onClick={() => setEnroll(null)}>Retry</Button>
        </div>
      ) : (
        <div className="space-y-4">
          {enroll.qr && <Qr code={enroll.qr} />}
          {enroll.secret && (
            <p className="text-center text-xs text-muted-foreground">
              Can’t scan? Enter this key:<br />
              <code className="mt-1 inline-block break-all rounded bg-muted px-2 py-1 font-mono text-foreground">{enroll.secret}</code>
            </p>
          )}
          <form action={doVerify} className="space-y-3">
            <input type="hidden" name="factorId" value={enroll.factorId ?? ""} />
            <CodeField label="6-digit code / कोड" />
            {verify.error && <Alert variant="error">{verify.error}</Alert>}
            <Button type="submit" className="w-full" disabled={verifying}>
              {verifying && <Spinner />} Finish / पूर्ण करें
            </Button>
          </form>
        </div>
      )}
    </>
  );
}
