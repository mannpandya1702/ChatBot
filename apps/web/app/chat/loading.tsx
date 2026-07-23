import { Spinner } from "@/lib/ui/misc";

export default function Loading() {
  return (
    <div className="flex min-h-dvh items-center justify-center bg-background">
      <Spinner className="h-6 w-6 text-primary" />
    </div>
  );
}
