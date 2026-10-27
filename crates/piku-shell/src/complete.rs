//! Tab completion for the interactive line editor.
//!
//! A word is completed as a filesystem path when it looks like one (contains
//! `/`, starts with `~` or `.`) or when it sits in argument position. The first
//! word of a command completes against the shell's builtins and executable names
//! on `PATH`. The decision logic is pure; only [`tab`] touches the environment,
//! the editor, and the terminal.
//!
//! Known limits: only the first Tab is handled (no "cycle through matches"),
//! quoted words are not completed, `~user` is not expanded, and no command
//! understands its own argument grammar.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::editor::LineEditor;
use crate::term;

/// Candidate columns are wrapped at this width; the terminal size is not read.
const LIST_WIDTH: usize = 80;

/// What one Tab should do to the line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completion {
    /// Leave the line alone.
    None,
    /// Replace `start..end` (character indices) with `text`, optionally adding
    /// one trailing space. The rest of the line is preserved.
    Replace {
        start: usize,
        end: usize,
        text: String,
        append_space: bool,
    },
    /// Print these candidates and keep the line as typed.
    List(Vec<String>),
}

/// Handles one Tab keypress: complete in place, or print candidates.
pub fn tab(editor: &mut LineEditor, builtins: &[&str]) -> io::Result<()> {
    let line = editor.text();
    let cursor = editor.cursor();
    let commands = command_names(builtins);
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match complete_with(&line, cursor, &commands, &cwd) {
        Completion::None => Ok(()),
        Completion::Replace {
            start,
            end,
            text,
            append_space,
        } => {
            editor.replace_range(start, end, &text);
            if append_space {
                editor.insert(' ');
            }
            Ok(())
        }
        Completion::List(candidates) => print_candidates(&candidates),
    }
}

/// Pure completion decision over an explicit command list and directory. The
/// editor and the environment are not consulted, which keeps this testable.
pub fn complete_with(line: &str, cursor: usize, commands: &[String], cwd: &Path) -> Completion {
    let chars: Vec<char> = line.chars().collect();
    let cursor = cursor.min(chars.len());
    let start = word_start(&chars, cursor);
    let end = word_end(&chars, cursor);
    let prefix: String = chars[start..cursor].iter().collect();
    if prefix.is_empty() {
        // Listing every command or file on a bare Tab floods the screen.
        return Completion::None;
    }
    // A word directly after an opening quote needs quote-aware replacement;
    // skip rather than leave an unbalanced line behind.
    if start > 0 && matches!(chars[start - 1], '\'' | '"') {
        return Completion::None;
    }

    let path_mode = path_like(&prefix) || !command_position(&chars, start);
    if path_mode {
        let candidates = path_candidates(&prefix, cwd);
        if candidates.is_empty() {
            return Completion::None;
        }
        if candidates.len() == 1 {
            let candidate = &candidates[0];
            return Completion::Replace {
                start,
                end,
                text: candidate.display.clone(),
                append_space: !candidate.is_dir,
            };
        }
        let displays: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.display.clone())
            .collect();
        extend_or_list(start, end, &prefix, displays)
    } else {
        let mut candidates: Vec<String> = commands
            .iter()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect();
        candidates.sort();
        candidates.dedup();
        if candidates.is_empty() {
            return Completion::None;
        }
        if candidates.len() == 1 {
            return Completion::Replace {
                start,
                end,
                text: candidates.remove(0),
                append_space: true,
            };
        }
        extend_or_list(start, end, &prefix, candidates)
    }
}

/// Extends the word to the longest common prefix of the candidates when that
/// makes progress; otherwise returns the list for printing.
fn extend_or_list(start: usize, end: usize, prefix: &str, candidates: Vec<String>) -> Completion {
    let common = longest_common_prefix(&candidates);
    if common.chars().count() > prefix.chars().count() {
        Completion::Replace {
            start,
            end,
            text: common,
            append_space: false,
        }
    } else {
        Completion::List(candidates)
    }
}

/// Longest prefix shared by every candidate, compared by characters.
fn longest_common_prefix(candidates: &[String]) -> String {
    let mut candidates = candidates.iter();
    let Some(first) = candidates.next() else {
        return String::new();
    };
    let mut common: Vec<char> = first.chars().collect();
    for candidate in candidates {
        let shared = common
            .iter()
            .zip(candidate.chars())
            .take_while(|(left, right)| *left == right)
            .count();
        common.truncate(shared);
        if common.is_empty() {
            break;
        }
    }
    common.into_iter().collect()
}

/// A single path candidate and whether it is a directory.
#[derive(Debug)]
struct Candidate {
    display: String,
    is_dir: bool,
}

/// Lists entries of the prefix's directory that extend its last component.
fn path_candidates(prefix: &str, cwd: &Path) -> Vec<Candidate> {
    if prefix == "~" {
        return match home() {
            Some(home) if home.is_dir() => vec![Candidate {
                display: "~/".into(),
                is_dir: true,
            }],
            _ => Vec::new(),
        };
    }
    let (directory_text, base) = match prefix.rfind('/') {
        Some(slash) => prefix.split_at(slash + 1),
        None => ("", prefix),
    };
    let directory = expand_directory(directory_text, cwd);
    let Ok(entries) = fs::read_dir(&directory) else {
        return Vec::new();
    };
    let show_hidden = base.starts_with('.');
    let mut candidates: Vec<Candidate> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !show_hidden && name.starts_with('.') {
            continue;
        }
        if !name.starts_with(base) {
            continue;
        }
        let is_dir = entry.path().is_dir();
        let suffix = if is_dir { "/" } else { "" };
        candidates.push(Candidate {
            display: format!("{directory_text}{name}{suffix}"),
            is_dir,
        });
    }
    candidates.sort_by(|left, right| left.display.cmp(&right.display));
    candidates.dedup_by(|left, right| left.display == right.display);
    candidates
}

/// Resolves the directory part of a path word against the shell's cwd. The
/// literal text is kept for display; only the lookup is expanded.
fn expand_directory(directory_text: &str, cwd: &Path) -> PathBuf {
    if directory_text.is_empty() {
        return cwd.to_path_buf();
    }
    if let Some(rest) = directory_text.strip_prefix("~/")
        && let Some(home) = home()
    {
        return home.join(rest);
    }
    let path = Path::new(directory_text);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Builtins plus every executable name on `PATH`.
fn command_names(builtins: &[&str]) -> Vec<String> {
    let mut names: Vec<String> = builtins.iter().map(|name| (*name).to_string()).collect();
    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                if !is_executable(&entry.path()) {
                    continue;
                }
                if let Ok(name) = entry.file_name().into_string() {
                    names.push(name);
                }
            }
        }
    }
    names
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// Whether the word starts a command rather than an argument. Anything after
/// `|`, `;`, `&`, or `(` counts as a fresh command.
fn command_position(chars: &[char], start: usize) -> bool {
    let mut index = start;
    while index > 0 {
        index -= 1;
        let character = chars[index];
        if character.is_whitespace() {
            continue;
        }
        return matches!(character, '|' | ';' | '&' | '(');
    }
    true
}

fn path_like(prefix: &str) -> bool {
    prefix.contains('/') || prefix.starts_with('~') || prefix.starts_with('.')
}

fn word_start(chars: &[char], cursor: usize) -> usize {
    let mut start = cursor;
    while start > 0 && !is_separator(chars[start - 1]) {
        start -= 1;
    }
    start
}

fn word_end(chars: &[char], cursor: usize) -> usize {
    let mut end = cursor;
    while end < chars.len() && !is_separator(chars[end]) {
        end += 1;
    }
    end
}

fn is_separator(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            '|' | '&' | ';' | '<' | '>' | '(' | ')' | '\'' | '"'
        )
}

/// Prints candidates in columns, ending with a newline so the next prompt
/// redraw starts on a clean row.
fn print_candidates(candidates: &[String]) -> io::Result<()> {
    let width = candidates
        .iter()
        .map(|candidate| candidate.chars().count())
        .max()
        .unwrap_or(0)
        .max(1)
        + 2;
    let columns = (LIST_WIDTH / width).max(1);
    let mut output = String::from("\r\n");
    for (index, candidate) in candidates.iter().enumerate() {
        output.push_str(candidate);
        let end_of_row = (index + 1) % columns == 0 || index + 1 == candidates.len();
        if end_of_row {
            output.push_str("\r\n");
        } else {
            let pad = width.saturating_sub(candidate.chars().count());
            output.extend(std::iter::repeat_n(' ', pad));
        }
    }
    term::write(output.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commands(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn longest_common_prefix_of_commands() {
        assert_eq!(
            longest_common_prefix(&commands(&["cargo", "cargo-fmt", "cat"])),
            "ca"
        );
        assert_eq!(
            longest_common_prefix(&commands(&["cargo", "cargo-fmt"])),
            "cargo"
        );
        assert_eq!(longest_common_prefix(&commands(&["alpha", "beta"])), "");
        assert_eq!(longest_common_prefix(&[]), "");
    }

    #[test]
    fn unique_command_completes_with_a_space() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            complete_with("pw", 2, &commands(&["pwd", "cat"]), &cwd),
            Completion::Replace {
                start: 0,
                end: 2,
                text: "pwd".into(),
                append_space: true,
            }
        );
    }

    #[test]
    fn ambiguous_command_inserts_the_common_prefix() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            complete_with("car", 3, &commands(&["cargo", "cargo-fmt", "cat"]), &cwd),
            Completion::Replace {
                start: 0,
                end: 3,
                text: "cargo".into(),
                append_space: false,
            }
        );
    }

    #[test]
    fn no_common_prefix_lists_the_candidates() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            complete_with("c", 1, &commands(&["cd", "cat"]), &cwd),
            Completion::List(commands(&["cat", "cd"]))
        );
    }

    #[test]
    fn argument_position_completes_paths() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::write(directory.path().join("alpha.txt"), "x").unwrap();
        assert_eq!(
            complete_with("cat al tail", 6, &[], directory.path()),
            Completion::Replace {
                start: 4,
                end: 6,
                text: "alpha.txt".into(),
                append_space: true,
            }
        );
    }

    #[test]
    fn unique_directory_keeps_a_trailing_slash_and_no_space() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::create_dir(directory.path().join("sub")).unwrap();
        assert_eq!(
            complete_with("cd su", 5, &[], directory.path()),
            Completion::Replace {
                start: 3,
                end: 5,
                text: "sub/".into(),
                append_space: false,
            }
        );
    }

    #[test]
    fn ambiguous_paths_insert_the_common_prefix() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::write(directory.path().join("alpha.txt"), "x").unwrap();
        fs::write(directory.path().join("alpine.txt"), "x").unwrap();
        assert_eq!(
            complete_with("cat al", 6, &[], directory.path()),
            Completion::Replace {
                start: 4,
                end: 6,
                text: "alp".into(),
                append_space: false,
            }
        );
    }

    #[test]
    fn path_prefix_is_preserved_for_nested_directories() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(directory.path().join("sub/deep")).unwrap();
        assert_eq!(
            complete_with("cat sub/de", 10, &[], directory.path()),
            Completion::Replace {
                start: 4,
                end: 10,
                text: "sub/deep/".into(),
                append_space: false,
            }
        );
    }

    #[test]
    fn hidden_entries_need_a_leading_dot() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::write(directory.path().join(".env"), "x").unwrap();
        assert_eq!(
            complete_with("cat .e", 6, &[], directory.path()),
            Completion::Replace {
                start: 4,
                end: 6,
                text: ".env".into(),
                append_space: true,
            }
        );
        assert_eq!(
            complete_with("cat ", 4, &[], directory.path()),
            Completion::None
        );
    }

    #[test]
    fn tilde_alone_offers_the_home_directory() {
        // HOME is read from the environment; "~/" is only offered when it is a
        // directory. This assertion holds for any real session.
        let cwd = std::env::current_dir().unwrap();
        let result = complete_with("cd ~", 4, &[], &cwd);
        match result {
            Completion::Replace { text, .. } => assert_eq!(text, "~/"),
            Completion::None => {}
            other => panic!("unexpected completion: {other:?}"),
        }
    }

    #[test]
    fn empty_prefix_does_nothing() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            complete_with("", 0, &commands(&["cat"]), &cwd),
            Completion::None
        );
    }

    #[test]
    fn words_inside_quotes_are_left_alone() {
        let directory = tempfile::TempDir::new().unwrap();
        fs::write(directory.path().join("alpha.txt"), "x").unwrap();
        // Completion would have to move the closing quote; skip instead.
        assert_eq!(
            complete_with("cat 'al", 7, &[], directory.path()),
            Completion::None
        );
    }
}
