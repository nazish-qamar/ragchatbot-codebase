# Frontend Changes: Dark/Light Theme Toggle

## Summary

Added a dark/light mode toggle button positioned in the top-right corner of the page, using sun/moon icons with smooth animations, localStorage persistence, and system preference detection.

## Files Changed

### `frontend/index.html`
- Added a `<button id="themeToggle">` element before the main container with:
  - Two inline SVG icons (sun and moon) from the Feather icon set
  - `aria-label` and `title` attributes for accessibility
  - Keyboard-navigable by default (native `<button>` element)

### `frontend/style.css`
- Added `[data-theme="light"]` CSS custom properties block that overrides all `--var` colors for light mode (background, surface, text, borders, shadows)
- Added `.theme-toggle` button styles: fixed positioning top-right, circular shape, hover/focus/active states
- Added icon visibility rules: moon icon shows in dark mode, sun icon shows in light mode
- Added `transition` declarations on key elements for smooth 0.3s color transitions when switching themes
- Added light-theme-specific overrides for code blocks (`background-color` adjustments)

### `frontend/script.js`
- Added `initTheme()` function: reads saved theme from `localStorage`, falls back to `prefers-color-scheme` media query, sets `data-theme` attribute on `<html>`
- Added `toggleTheme()` function: toggles `data-theme` between `"light"` and `"dark"`, persists choice to `localStorage`
- `initTheme()` is called immediately (before `DOMContentLoaded`) to prevent theme flash on page load
- Theme toggle click listener is registered in the `DOMContentLoaded` handler

## Design Decisions

- **`data-theme` attribute on `<html>`**: Allows CSS to cascade from the root, keeping the toggle mechanism decoupled from individual component styles
- **`localStorage` persistence**: User's theme choice survives page reloads and browser restarts
- **System preference fallback**: Respects `prefers-color-scheme: light` when no saved preference exists; defaults to the existing dark theme otherwise
- **Native `<button>` element**: Ensures keyboard accessibility (focusable, Enter/Space activation) without extra ARIA work
- **CSS transitions on theme-aware elements**: Provides a smooth 0.3s crossfade when toggling rather than an abrupt color change
