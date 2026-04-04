/// Parse key=value strings and CSV lines.

/// Parse a `key=value` string. Returns None if `=` is absent.
pub fn parse_kv(s: &str) -> Option<(&str, &str)> {
    let pos = s.find('=')?;
    Some((&s[..pos], &s[pos + 1..]))
}

/// Split a CSV line into fields.
///
/// BUG: consecutive commas produce empty strings but the function
/// silently drops them instead of returning empty fields.
/// Example: "a,,b" should yield ["a", "", "b"] but returns ["a", "b"].
///
/// Scenario: ask piku to find the bug and fix it.
pub fn split_csv(line: &str) -> Vec<&str> {
    line.split(',').filter(|s| !s.is_empty()).collect() // BUG: filter drops empty fields
}

/// Extract the value from a `--key=value` CLI flag, if present.
pub fn parse_cli_flag<'a>(arg: &'a str, flag: &str) -> Option<&'a str> {
    let prefix = format!("--{}=", flag);
    if arg.starts_with(&prefix) {
        Some(&arg[prefix.len()..])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_normal() {
        assert_eq!(parse_kv("key=value"), Some(("key", "value")));
    }

    #[test]
    fn parse_kv_missing_eq() {
        assert_eq!(parse_kv("noequals"), None);
    }

    #[test]
    fn split_csv_simple() {
        assert_eq!(split_csv("a,b,c"), vec!["a", "b", "c"]);
    }

    // MISSING: test for consecutive commas that would expose the bug
    // Scenario: ask piku to add a failing test, then fix split_csv.
}
