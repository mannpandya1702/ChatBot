import Link from "next/link";
import { Button } from "@/lib/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-background px-4 text-center">
      <div className="text-5xl" aria-hidden="true">🧭</div>
      <div>
        <h1 className="text-xl font-semibold">Page not found</h1>
        <p className="mt-1 text-sm text-muted-foreground">यह पृष्ठ उपलब्ध नहीं है। The page you’re looking for doesn’t exist.</p>
      </div>
      <Link href="/chat"><Button>Go to chat</Button></Link>
    </main>
  );
}
