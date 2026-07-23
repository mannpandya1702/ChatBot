/** Lightweight script-based language tag for analytics + NOT_FOUND selection. */
export type Lang = "hi" | "en" | "hinglish";

const DEVANAGARI = /[ऀ-ॿ]/g;
const LATIN = /[A-Za-z]/g;

export function detectLanguage(text: string): Lang {
  const deva = (text.match(DEVANAGARI) ?? []).length;
  const latin = (text.match(LATIN) ?? []).length;
  const total = deva + latin;
  if (total === 0) return "en";
  const ratio = deva / total;
  if (ratio > 0.6) return "hi";
  if (ratio > 0.1) return "hinglish";
  return "en";
}
