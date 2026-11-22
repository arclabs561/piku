//! Word expansion: a leading `~`, `$VAR`, `${VAR}`, `$?`, and field splitting.
//!
//! Splitting is the POSIX rule for unquoted expansions: the value is split on
//! runs of blanks, quotes protect their text, and an unquoted expansion that
//! produces nothing removes its word. Globs are not expanded, so `*` stays a
//! literal asterisk.

use std::collections::BTreeMap;

use crate::syntax::{Expansion, Part, Word};

/// Expands each word into the argument fields it produces.
pub fn words(words: &[Word], vars: &BTreeMap<String, String>, status: i32) -> Vec<String> {
    words
        .iter()
        .flat_map(|word| fields(word, vars, status))
        .collect()
}

fn fields(word: &Word, vars: &BTreeMap<String, String>, status: i32) -> Vec<String> {
    let mut segments = Vec::with_capacity(word.parts().len() + 1);
    if word.tilde() {
        segments.push(Segment::Literal(home(vars)));
    }
    for part in word.parts() {
        match part {
            Part::Literal(text) => segments.push(Segment::Literal(text.clone())),
            Part::Quoted(expansion) => {
                segments.push(Segment::Literal(value(expansion, vars, status)));
            }
            Part::Unquoted(expansion) => {
                segments.push(Segment::Split(value(expansion, vars, status)));
            }
        }
    }
    split(&segments)
}

/// A piece of a word, before splitting decides where the field boundaries go.
enum Segment {
    /// Never split, and always keeps its field alive even when empty.
    Literal(String),
    /// Split on blanks; contributes no field when it expands to nothing.
    Split(String),
}

fn split(segments: &[Segment]) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_field = false;
    // An empty quoted run wants a field, but joins one that is already open.
    let mut pending_empty = false;

    for segment in segments {
        match segment {
            Segment::Literal(text) => {
                if text.is_empty() {
                    if !in_field {
                        pending_empty = true;
                    }
                } else {
                    field.push_str(text);
                    in_field = true;
                    pending_empty = false;
                }
            }
            Segment::Split(text) => {
                for character in text.chars() {
                    if is_blank(character) {
                        if in_field {
                            fields.push(std::mem::take(&mut field));
                            in_field = false;
                        }
                    } else {
                        pending_empty = false;
                        field.push(character);
                        in_field = true;
                    }
                }
            }
        }
    }

    if in_field {
        fields.push(field);
    } else if pending_empty {
        fields.push(String::new());
    }
    fields
}

fn value(expansion: &Expansion, vars: &BTreeMap<String, String>, status: i32) -> String {
    match expansion {
        Expansion::Variable(name) => vars.get(name).cloned().unwrap_or_default(),
        Expansion::Status => status.to_string(),
    }
}

fn home(vars: &BTreeMap<String, String>) -> String {
    vars.get("HOME").cloned().unwrap_or_else(|| "~".to_string())
}

fn is_blank(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::parse;

    fn vars(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn expand(line: &str, vars: &BTreeMap<String, String>, status: i32) -> Vec<String> {
        let program = parse(line).expect("line parses");
        let mut fields = Vec::new();
        for item in program.items {
            for command in item.pipeline.commands {
                fields.extend(words(&command.words, vars, status));
            }
        }
        fields
    }

    #[test]
    fn expands_variables_and_status() {
        let vars = vars(&[("NAME", "wish"), ("EMPTY", "")]);
        assert_eq!(expand("echo $NAME", &vars, 0), ["echo", "wish"]);
        assert_eq!(expand("echo ${NAME}", &vars, 0), ["echo", "wish"]);
        assert_eq!(expand("echo $?/${NAME}", &vars, 3), ["echo", "3/wish"]);
        assert_eq!(expand("echo \"$?\"", &vars, 4), ["echo", "4"]);
        assert_eq!(expand("echo $MISSING", &vars, 0), ["echo"]);
    }

    #[test]
    fn unquoted_expansions_split_on_blanks() {
        let vars = vars(&[("LIST", "a b\tc")]);
        assert_eq!(expand("$LIST", &vars, 0), ["a", "b", "c"]);
        assert_eq!(
            expand("pre $LIST post", &vars, 0),
            ["pre", "a", "b", "c", "post"]
        );
        assert_eq!(expand("x$LIST", &vars, 0), ["xa", "b", "c"]);
    }

    #[test]
    fn quotes_protect_text_and_keep_empty_fields() {
        let vars = vars(&[("LIST", "a b"), ("EMPTY", "")]);
        assert_eq!(expand("\"$LIST\"", &vars, 0), ["a b"]);
        assert_eq!(expand("'$LIST'", &vars, 0), ["$LIST"]);
        assert_eq!(expand("\"\"", &vars, 0), [""]);
        assert_eq!(expand("$EMPTY", &vars, 0), Vec::<String>::new());
        assert_eq!(expand("\"$EMPTY\"", &vars, 0), [""]);
        // A quoted empty run joins the field being built instead of adding one.
        assert_eq!(expand("$LIST\"\"", &vars, 0), ["a", "b"]);
        assert_eq!(expand("\"\"$LIST", &vars, 0), ["a", "b"]);
    }

    #[test]
    fn tilde_expands_from_home() {
        let vars = vars(&[("HOME", "/home/wish")]);
        assert_eq!(expand("~", &vars, 0), ["/home/wish"]);
        assert_eq!(expand("~/src", &vars, 0), ["/home/wish/src"]);
        assert_eq!(expand("~user", &vars, 0), ["~user"]);
        assert_eq!(expand("a~b", &vars, 0), ["a~b"]);
        let without_home = BTreeMap::new();
        assert_eq!(expand("~", &without_home, 0), ["~"]);
    }

    #[test]
    fn globs_stay_literal() {
        let vars = BTreeMap::new();
        assert_eq!(expand("echo *.rs", &vars, 0), ["echo", "*.rs"]);
    }
}
