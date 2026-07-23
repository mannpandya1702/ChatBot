"use client";
import { useActionState } from "react";
import { Button } from "@/lib/ui/button";
import { Input } from "@/lib/ui/input";
import { Label, Alert, Spinner } from "@/lib/ui/misc";
import { signInAction, verifyMfaAction, type LoginState } from "./actions";

const empty: LoginState = {};

export function LoginForm() {
  const [signIn, doSignIn, signingIn] = useActionState(signInAction, empty);
  const [mfa, doVerify, verifying] = useActionState(verifyMfaAction, empty);
  const onMfaStep = signIn.step === "mfa";

  return onMfaStep ? (
    <form action={doVerify} className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor="code">Authenticator code / प्रमाणक कोड</Label>
        <Input
          id="code"
          name="code"
          inputMode="numeric"
          autoComplete="one-time-code"
          maxLength={6}
          placeholder="123456"
          autoFocus
          required
        />
        <p className="text-xs text-muted-foreground">
          Enter the 6-digit code from your authenticator app.
        </p>
      </div>
      {mfa.error && <Alert variant="error">{mfa.error}</Alert>}
      <Button type="submit" className="w-full" disabled={verifying}>
        {verifying && <Spinner />} Verify / सत्यापित करें
      </Button>
    </form>
  ) : (
    <form action={doSignIn} className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor="serviceNumber">Service number / सेवा संख्या</Label>
        <Input id="serviceNumber" name="serviceNumber" autoComplete="username" autoCapitalize="characters" autoFocus required />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="password">Password / पासवर्ड</Label>
        <Input id="password" name="password" type="password" autoComplete="current-password" required />
      </div>
      {signIn.error && <Alert variant="error">{signIn.error}</Alert>}
      <Button type="submit" className="w-full" disabled={signingIn}>
        {signingIn && <Spinner />} Sign in / साइन इन करें
      </Button>
    </form>
  );
}
