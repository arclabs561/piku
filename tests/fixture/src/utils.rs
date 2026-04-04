/// Utility formatting functions.
// NOTE: format_output is intentionally undocumented.
// Scenario: ask piku to add doc comments to all public functions.

pub fn format_output(label: &str, value: i32) -> String {
    format!("{}: {}", label, value)
}

pub fn pluralise(n: i32, singular: &str, plural: &str) -> String {
    if n == 1 {
        format!("{n} {singular}")
    } else {
        format!("{n} {plural}")
    }
}

/// Pad a string to a minimum width with spaces on the right.
pub fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - s.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pluralise_singular() {
        assert_eq!(pluralise(1, "item", "items"), "1 item");
    }

    #[test]
    fn pluralise_plural() {
        assert_eq!(pluralise(3, "item", "items"), "3 items");
    }
}
