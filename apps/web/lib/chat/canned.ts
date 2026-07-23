import type { AppSettings, Bilingual } from "./types";
import type { Lang } from "../lang";

/** Bilingual greeting reply — no retrieval (spec §3). */
export const GREETING: Bilingual = {
  en: "Namaste. I am Sainik Sahayak. Aap seva niyam, pension, chhutti, ECHS, AGIF ya welfare se judi jaankari puchh sakte hain.",
  hi: "नमस्ते। मैं सैनिक सहायक हूँ। आप सेवा नियम, पेंशन, छुट्टी, ECHS, AGIF या कल्याण से जुड़ी जानकारी पूछ सकते हैं।",
};

/** Polite scope message for injection/jailbreak attempts (spec §3, §8). */
export const INJECTION_SCOPE: Bilingual = {
  en: "I can only provide official information from the Army knowledge base. Please ask a question about service matters such as leave, pension, ECHS, AGIF, or welfare.",
  hi: "मैं केवल सेना ज्ञानकोश से आधिकारिक जानकारी दे सकता हूँ। कृपया सेवा संबंधी प्रश्न पूछें, जैसे छुट्टी, पेंशन, ECHS, AGIF या कल्याण।",
};

function pick(b: Bilingual, lang: Lang): string {
  return lang === "hi" ? b.hi : b.en;
}

/** NOT_FOUND message with the escalation contact appended (spec §7). */
export function notFoundMessage(settings: AppSettings, lang: Lang): string {
  const base = pick(settings.notFound, lang);
  const esc = pick(settings.escalationContact, lang).trim();
  return esc ? `${base} ${esc}` : base;
}

export function greeting(lang: Lang): string {
  return pick(GREETING, lang);
}

export function injectionScope(lang: Lang): string {
  return pick(INJECTION_SCOPE, lang);
}
