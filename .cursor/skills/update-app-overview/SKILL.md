---
name: update-app-overview
description: Sync .cursor/rules/juggle-overview.mdc with the current codebase. Use only when the user explicitly invokes /update-app-overview or asks to refresh the app overview rule.
disable-model-invocation: true
---

# Update app overview rule

Sync `.cursor/rules/juggle-overview.mdc` with the current state of the JuggleJs project.

## When to run

Only when explicitly requested by the user. Do not update overview after unrelated code changes.

## Steps

1. Read: `js/app.js`, `index.html`, `test-juggle-video.html`, `README.md`, current `juggle-overview.mdc`.
2. Compare overview vs code. Update only sections that are wrong or missing:
   - Architecture (SPA, vanilla JS, file layout)
   - Juggle counting pipeline
   - Debug tooling (highlighter, snake plot, timing stats, test DEBUG mode)
   - Test API and events
   - Notable user-facing features (e.g. voice count)
3. Do NOT add: changelog, line-by-line code, implementation details better left in source.
4. Keep the file concise (target under ~60 lines). Remove outdated bullets.
5. Do not modify `juggle-conventions.mdc` unless architecture affects coding rules.
6. Show the user a short summary of what changed.
7. Commit this changes even without explicit user's request.

## Output rules

- Preserve YAML frontmatter: `alwaysApply: true`
- English in the rule file body
- One fact per bullet; no duplication of README prose
