// Flat ESLint config (ESLint 10). Lean and correctness-focused: the compiler
// (`tsc --noEmit`) already owns types, so lint adds the checks types don't —
// unused code, unsafe `any`, and the rules of hooks. Stylistic noise is dialled
// down so `npm run lint` stays a signal, not a chore.
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  {
    // Generated / vendored / non-source output — never linted.
    ignores: [
      "node_modules/**",
      ".next/**",
      "next-env.d.ts",
      "public/**",
      "coverage/**",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // Wire react-hooks manually — the plugin's shipped presets still use the
    // legacy string-array `plugins` shape that ESLint 10 flat config rejects.
    // These two rules are the whole value: hooks called unconditionally, deps
    // arrays kept honest.
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
  {
    rules: {
      // Flag unused code, but let an intentional `_`-prefix opt out (common for
      // required-but-unused args and catch bindings).
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      // `any` is a smell, not a build-breaker — surface it without failing CI.
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },
  {
    // Node-context scripts and configs (plain .mjs): declare the Node globals so
    // no-undef doesn't fire on process/Buffer/console. (.mts/.ts get these from
    // TypeScript, which also turns no-undef off.)
    files: ["scripts/**", "*.config.{js,mjs,ts}", "*.mjs"],
    languageOptions: {
      globals: {
        process: "readonly",
        Buffer: "readonly",
        console: "readonly",
        URL: "readonly",
        fetch: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        __dirname: "readonly",
        __filename: "readonly",
        module: "writable",
        require: "readonly",
        global: "readonly",
      },
    },
    rules: {
      "no-console": "off",
    },
  },
  {
    // Test scaffolding legitimately casts partial mocks — `any` is expected here.
    files: ["**/*.test.ts", "**/*.test.tsx"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
);
