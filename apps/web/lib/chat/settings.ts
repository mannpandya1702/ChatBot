import { serviceClient } from "../supabase/service";
import { env } from "../env";
import type { AppSettings, Bilingual } from "./types";

function bilingual(v: unknown): Bilingual {
  const o = (v ?? {}) as Record<string, unknown>;
  return { hi: String(o.hi ?? ""), en: String(o.en ?? "") };
}

/** Load admin-editable settings (spec §4 app_settings). */
export async function loadSettings(): Promise<AppSettings> {
  const svc = serviceClient();
  const { data, error } = await svc.from("app_settings").select("key,value");
  if (error) throw new Error(`load settings: ${error.message}`);
  const map = new Map(
    (data as { key: string; value: unknown }[]).map((r) => [r.key, r.value]),
  );
  const threshold = map.get("rerank_refusal_threshold");
  return {
    notFound: bilingual(map.get("not_found_message")),
    escalationContact: bilingual(map.get("escalation_contact")),
    rerankRefusalThreshold:
      typeof threshold === "number" ? threshold : env.rerankRefusalThreshold,
  };
}
