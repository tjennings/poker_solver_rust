# Frontend Refactor: Discord-Like Left Rail Navigation

**Date**: 2026-03-05
**Status**: Approved

## Summary

Refactor the Tauri frontend from tab-based navigation to a Discord-like left rail with icon navigation. Apply a modern dark theme with gradients and shadows.

## Layout

```
+--------+------------------------------------+
| Rail   |  Right Pane                        |
| 64px   |  (fills remaining width)           |
|        |                                    |
| [icon] |  Explore / Train / Arena / Settings|
| [icon] |                                    |
| [icon] |                                    |
|        |                                    |
| [gear] |                                    |
+--------+------------------------------------+
```

- Left rail: 64px fixed width, full viewport height
- Icons: 40px buttons with tooltip on hover
- Active state: 3px left pill in accent color, icon tinted
- Settings icon pushed to bottom via `margin-top: auto`
- Right pane: fills remaining width, scrollable, no max-width

## Views

| Rail Icon | View | Content |
|-|-|-|
| Compass SVG | Explore | Current Explorer minus hamburger menu |
| Flask SVG | Train | Stub: "Training configuration coming soon" |
| Swords SVG | Arena | Current Simulator unchanged |
| Gear SVG | Settings | Stub: "Settings coming soon" |

## Explore: Load Dataset Flow

- **No dataset loaded**: Action strip shows single "Load Dataset" action card (folder icon + text)
- **Click**: triggers file picker dialog (Tauri `open()` or `window.prompt()` fallback)
- **After load**: normal Explorer behavior
- **Hamburger menu**: removed entirely

## Visual Theme

- **Body background**: gradient `#0f0f1e` to `#1a1a2e`
- **Rail**: `#0a0a18`, `box-shadow: 2px 0 12px rgba(0,0,0,0.4)`
- **Cards/panels**: `#14142a` to `#1a1a3a` gradient, `box-shadow: 0 2px 8px rgba(0,0,0,0.3)`, border `rgba(255,255,255,0.06)`
- **Active rail icon**: 3px left pill `#00d9ff`
- **Primary accent**: `#00d9ff`
- **Secondary accent**: `#7c3aed` (purple hover glows)
- **Text**: `#eee` primary, `#888` secondary, `#555` tertiary
