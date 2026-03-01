# Word Wrap Architecture in VectorSmite

## Problem Statement

Ratatui's TUI framework has limited built-in text wrapping support. While `Paragraph::wrap()` exists, it has constraints:

1. **Wrapping only works on Spans within Lines** - pre-built `Line` objects are treated as atomic units
2. **No wrap state exposure** - after wrapping, you can't query how many visual lines resulted
3. **Character boundary limitation** - wrapping only occurs at word boundaries, not character boundaries

This creates challenges when displaying long text (like vec2text inversions) in a scrollable panel.

## Failed Approaches

### Attempt 1: Just add `.wrap()`
```rust
Paragraph::new(Text::from(lines))
    .wrap(Wrap { trim: true })
```
**Why it failed:** Lines were pre-built with full text content. Paragraph's wrap doesn't split existing Lines.

### Attempt 2: Character chunking with dynamic width
```rust
let wrap_width = panel_width.saturating_sub(11);
let chars: Vec<char> = text.chars().collect();
for chunk in chars.chunks(wrap_width) {
    // build Line from chunk
}
```
**Why it failed:** `wrap_width` calculation was unreliable across different terminal sizes.

### Attempt 3: Byte-based slicing
```rust
let break_at = remaining[..wrap_width].rfind(' ').unwrap_or(wrap_width);
```
**Why it failed:** Rust strings use byte indices, not character indices. Multi-byte UTF-8 characters caused panics or incorrect splits.

## Working Solution

### The `wrap_text` Helper Function

Located at `src/main.rs:1440`:

```rust
fn wrap_text(text: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut current_len = 0;

    for word in text.split_whitespace() {
        if current_len == 0 {
            result.push_str(word);
            current_len = word.len();
        } else if current_len + 1 + word.len() <= max_width {
            result.push(' ');
            result.push_str(word);
            current_len += 1 + word.len();
        } else {
            result.push('\n');
            result.push_str(word);
            current_len = word.len();
        }
    }

    result
}
```

**How it works:**
1. Split text into words using `split_whitespace()` (handles any whitespace)
2. Accumulate words into current line
3. When adding a word would exceed `max_width`, insert `\n` and start new line
4. Returns a single String with embedded newlines

### Usage in Inversions Panel

```rust
let wrap_at = 45usize;  // Fixed, reliable width

let wrapped = wrap_text(&inv.inverted, wrap_at);
for line in wrapped.lines() {
    all_lines.push(Line::from(vec![
        Span::styled("  ", Style::default()),
        Span::styled(line.to_string(), Style::default().fg(score_color)),
    ]));
}
```

**Key design decisions:**
1. **Fixed wrap width (45 chars)** - More reliable than dynamic calculation
2. **Word-based wrapping** - Preserves readability, avoids mid-word breaks
3. **Pre-wrap before building Lines** - Each resulting line is a separate `Line` object
4. **Paragraph::wrap() as fallback** - Still enabled for edge cases

### Usage in Source Overview

For the source overview panel when showing inversions inline:

```rust
let wrapped = wrap_text(&inversion.inverted, 50);
append_text(&mut lines, &mut spans, &wrapped, inv_style);
```

The `append_text` function already handles `\n` characters by creating new Lines, so the pre-wrapped text displays correctly.

## Scrolling Compatibility

Because we pre-wrap and create discrete `Line` objects, scrolling calculations remain accurate:

```rust
let total_lines = all_lines.len();  // Accurate count after wrapping
let visible_height = body[0].height.saturating_sub(2) as usize;
let max_scroll = total_lines.saturating_sub(visible_height);
```

## Key Lessons

1. **Pre-process text before building ratatui structures** - Don't rely on widget-level wrapping
2. **Use word-based splitting** - `split_whitespace()` is robust and Unicode-safe
3. **Fixed widths are more reliable** - Dynamic width calculation has edge cases
4. **Embed newlines in the source string** - Let existing line-handling code do its job
5. **Test with long text** - vec2text outputs can be surprisingly long

## References

- [Ratatui Paragraph docs](https://docs.rs/ratatui/latest/ratatui/widgets/struct.Paragraph.html)
- [RFC: Text Wrapping Design #293](https://github.com/ratatui/ratatui/issues/293)
- [Ratatui Paragraph recipe](https://ratatui.rs/recipes/widgets/paragraph/)
