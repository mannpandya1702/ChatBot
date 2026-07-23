import type { HTMLAttributes, LabelHTMLAttributes, ReactNode } from "react";
import { cn } from "./cn";

export function Label({ className, ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("text-sm font-medium text-foreground", className)} {...props} />;
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-lg border border-border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  );
}

export function Alert({
  variant = "info",
  className,
  children,
}: {
  variant?: "info" | "error" | "success";
  className?: string;
  children: ReactNode;
}) {
  const styles = {
    info: "border-border bg-muted text-foreground",
    error: "border-destructive/40 bg-destructive/10 text-destructive",
    success: "border-primary/40 bg-primary/10 text-primary",
  }[variant];
  return (
    <div role={variant === "error" ? "alert" : "status"} className={cn("rounded-md border px-3 py-2 text-sm", styles, className)}>
      {children}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("h-4 w-4 animate-spin", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
    </svg>
  );
}
