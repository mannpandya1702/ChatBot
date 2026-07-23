import { z } from "zod";

/** Chat request validation (spec §3: input cap 2000 chars). */
export const chatRequestSchema = z.object({
  message: z.string().min(1).max(2000),
  conversationId: z.string().uuid().optional(),
});

export type ChatRequestBody = z.infer<typeof chatRequestSchema>;
