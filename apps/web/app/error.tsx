"use client";
import { Button } from "@/lib/ui/button";

export default function Error({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-background px-4 text-center">
      <div className="text-5xl" aria-hidden="true">⚠️</div>
      <div>
        <h1 className="text-xl font-semibold">Something went wrong</h1>
        <p className="mt-1 text-sm text-muted-foreground">कुछ गड़बड़ हुई। Please try again.</p>
      </div>
      <Button onClick={reset}>Retry</Button>
    </main>
  );
}
