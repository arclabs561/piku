//! Tokenizer and parser for the native command language (M2/M3).
//!
//! Supported now: words made of unquoted text, single quotes, double quotes,
//! backslash escapes, `$VAR`, `${VAR}`, `$?`, and a leading `~`; comments that
//! start at a token boundary; the separators `;`, `&&`, and `||`; pipelines
//! `|`; redirections `<`, `>`, `>>`; and `&` after a pipeline to run it in the
//! background.
//!
//! `&` marks the pipeline it follows as asynchronous. This subset does not give
//! `&` POSIX's whole-and-or-list scope, so `a && b &` runs `a` in the
//! foreground first and then background `b`; use `(a && b) &` in a real shell.
//!
//! Deliberately absent: globbing, command substitution, arithmetic, functions,
//! aliases, and here-documents. The parser rejects what it cannot honour
//! instead of guessing, so a line either runs as written or does not run at
//! all.

/// One lexical piece of a word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    /// Text taken literally: unquoted plain characters, a single-quoted run, or
    /// an escaped character.
    Literal(String),
    /// An expansion outside quotes, subject to field splitting.
    Unquoted(Expansion),
    /// An expansion inside double quotes, never split.
    Quoted(Expansion),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expansion {
    /// `$NAME` or `${NAME}`.
    Variable(String),
    /// `$?`: the previous command's exit status.
    Status,
}

/// A word before expansion. An empty word is a real word only when it came
/// from quotes or escapes, which the parts record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Word {
    parts: Vec<Part>,
    /// The word began with an unquoted `~`, which expands to `$HOME`.
    tilde: bool,
}

impl Word {
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    pub fn tilde(&self) -> bool {
        self.tilde
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedirectKind {
    Write,
    Append,
    Read,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redirect {
    pub kind: RedirectKind,
    pub target: Word,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub words: Vec<Word>,
    pub redirects: Vec<Redirect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pipeline {
    pub commands: Vec<Command>,
}

/// How an item joins the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connector {
    /// First item of the line.
    First,
    /// `;`: run unconditionally.
    Sequence,
    /// `&&`: run only when the previous status is zero.
    And,
    /// `||`: run only when the previous status is nonzero.
    Or,
}

impl Connector {
    fn text(self) -> &'static str {
        match self {
            Self::First => "",
            Self::Sequence => ";",
            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub connector: Connector,
    pub pipeline: Pipeline,
    /// The pipeline ended with `&` and runs in the background.
    pub background: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Parses one command line. The whole line is validated before anything runs.
pub fn parse(line: &str) -> Result<Program, String> {
    Parser {
        tokens: tokenize(line)?,
        index: 0,
    }
    .program()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Semicolon,
    And,
    Or,
    Ampersand,
    Pipe,
    Write,
    Append,
    Read,
}

impl Op {
    fn text(self) -> &'static str {
        match self {
            Self::Semicolon => ";",
            Self::And => "&&",
            Self::Or => "||",
            Self::Ampersand => "&",
            Self::Pipe => "|",
            Self::Write => ">",
            Self::Append => ">>",
            Self::Read => "<",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Word(Word),
    Op(Op),
}

fn tokenize(line: &str) -> Result<Vec<Token>, String> {
    let characters: Vec<char> = line.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while let Some(&character) = characters.get(index) {
        match character {
            ' ' | '\t' | '\r' | '\n' => index += 1,
            // A comment starts only where a word could start, so `a#b` keeps
            // its hash.
            '#' => break,
            '&' => {
                if characters.get(index + 1) == Some(&'&') {
                    tokens.push(Token::Op(Op::And));
                    index += 2;
                } else {
                    tokens.push(Token::Op(Op::Ampersand));
                    index += 1;
                }
            }
            '|' => {
                if characters.get(index + 1) == Some(&'|') {
                    tokens.push(Token::Op(Op::Or));
                    index += 2;
                } else {
                    tokens.push(Token::Op(Op::Pipe));
                    index += 1;
                }
            }
            ';' => {
                tokens.push(Token::Op(Op::Semicolon));
                index += 1;
            }
            '>' => {
                if characters.get(index + 1) == Some(&'>') {
                    tokens.push(Token::Op(Op::Append));
                    index += 2;
                } else {
                    tokens.push(Token::Op(Op::Write));
                    index += 1;
                }
            }
            '<' => {
                if characters.get(index + 1) == Some(&'<') {
                    return Err("here-documents are not implemented yet".into());
                }
                tokens.push(Token::Op(Op::Read));
                index += 1;
            }
            _ => {
                let (word, next) = scan_word(&characters, index)?;
                tokens.push(Token::Word(word));
                index = next;
            }
        }
    }
    Ok(tokens)
}

fn scan_word(characters: &[char], mut index: usize) -> Result<(Word, usize), String> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut tilde = false;
    let mut at_start = true;

    while let Some(&character) = characters.get(index) {
        if is_word_break(character) {
            break;
        }
        match character {
            '\'' => {
                flush(&mut parts, &mut literal);
                let (text, next) = single_quoted(characters, index)?;
                parts.push(Part::Literal(text));
                index = next;
            }
            '"' => {
                flush(&mut parts, &mut literal);
                let (quoted, next) = double_quoted(characters, index)?;
                parts.extend(quoted);
                index = next;
            }
            '\\' => {
                let Some(&escaped) = characters.get(index + 1) else {
                    return Err("trailing backslash".into());
                };
                literal.push(escaped);
                index += 2;
            }
            '$' => {
                flush(&mut parts, &mut literal);
                match expansion(characters, index, Quote::Unquoted)? {
                    (Some(part), next) => {
                        parts.push(part);
                        index = next;
                    }
                    // A `$` that starts no expansion stays literal.
                    (None, next) => {
                        literal.push('$');
                        index = next;
                    }
                }
            }
            '~' if at_start => {
                // Only a leading `~` that ends the word or prefixes a path is
                // the home directory; `~user` stays as written.
                let next = characters.get(index + 1).copied();
                match next {
                    None => {
                        tilde = true;
                        index += 1;
                    }
                    Some('/') => {
                        tilde = true;
                        index += 1;
                    }
                    Some(next) if is_word_break(next) => {
                        tilde = true;
                        index += 1;
                    }
                    Some(_) => {
                        literal.push('~');
                        index += 1;
                    }
                }
            }
            _ => {
                literal.push(character);
                index += 1;
            }
        }
        at_start = false;
    }
    flush(&mut parts, &mut literal);
    Ok((Word { parts, tilde }, index))
}

fn single_quoted(characters: &[char], start: usize) -> Result<(String, usize), String> {
    let mut text = String::new();
    let mut index = start + 1;
    loop {
        match characters.get(index) {
            None => return Err("unclosed single quote".into()),
            Some('\'') => return Ok((text, index + 1)),
            Some(&character) => {
                text.push(character);
                index += 1;
            }
        }
    }
}

/// Scans a double-quoted run, which may contain expansions.
fn double_quoted(characters: &[char], start: usize) -> Result<(Vec<Part>, usize), String> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut index = start + 1;
    loop {
        match characters.get(index) {
            None => return Err("unclosed double quote".into()),
            Some('"') => {
                flush(&mut parts, &mut literal);
                // `""` is one empty field, not a word with no parts.
                if parts.is_empty() {
                    parts.push(Part::Literal(String::new()));
                }
                return Ok((parts, index + 1));
            }
            Some('\\') => match characters.get(index + 1) {
                None => return Err("unclosed double quote".into()),
                Some(&escaped) if matches!(escaped, '"' | '\\' | '$' | '`') => {
                    literal.push(escaped);
                    index += 2;
                }
                // Any other backslash is literal, as POSIX specifies.
                Some(_) => {
                    literal.push('\\');
                    index += 1;
                }
            },
            Some('$') => {
                flush(&mut parts, &mut literal);
                match expansion(characters, index, Quote::Quoted)? {
                    (Some(part), next) => {
                        parts.push(part);
                        index = next;
                    }
                    (None, next) => {
                        literal.push('$');
                        index = next;
                    }
                }
            }
            Some(&character) => {
                literal.push(character);
                index += 1;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Quote {
    Unquoted,
    Quoted,
}

/// Scans `$?`, `$NAME`, or `${NAME}` starting at `start`. `None` means the
/// dollar sign is literal and the caller keeps it.
fn expansion(
    characters: &[char],
    start: usize,
    quote: Quote,
) -> Result<(Option<Part>, usize), String> {
    let wrap = |expansion| match quote {
        Quote::Unquoted => Part::Unquoted(expansion),
        Quote::Quoted => Part::Quoted(expansion),
    };
    match characters.get(start + 1) {
        Some('?') => Ok((Some(wrap(Expansion::Status)), start + 2)),
        Some('{') => {
            let mut end = start + 2;
            while characters
                .get(end)
                .is_some_and(|&character| character != '}')
            {
                end += 1;
            }
            if characters.get(end).is_none() {
                return Err("unclosed '${'".into());
            }
            let inner: String = characters[start + 2..end].iter().collect();
            let expansion = if inner == "?" {
                Expansion::Status
            } else if is_identifier(&inner) {
                Expansion::Variable(inner)
            } else {
                return Err(format!("bad substitution: ${{{inner}}}"));
            };
            Ok((Some(wrap(expansion)), end + 1))
        }
        Some(&next) if is_name_start(next) => {
            let mut end = start + 1;
            while characters.get(end).is_some_and(|&next| is_name_char(next)) {
                end += 1;
            }
            let name: String = characters[start + 1..end].iter().collect();
            Ok((Some(wrap(Expansion::Variable(name))), end))
        }
        // `$`, `$@`, `$1`, and the like are not expansions in this subset.
        _ => Ok((None, start + 1)),
    }
}

/// Whether a name is a usable variable name.
pub fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    match characters.next() {
        Some(first) if is_name_start(first) => characters.all(is_name_char),
        _ => false,
    }
}

fn is_name_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_'
}

fn is_name_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

/// Whitespace and operators end a word.
fn is_word_break(character: char) -> bool {
    matches!(
        character,
        ' ' | '\t' | '\r' | '\n' | ';' | '&' | '|' | '>' | '<'
    )
}

fn flush(parts: &mut Vec<Part>, literal: &mut String) {
    if !literal.is_empty() {
        parts.push(Part::Literal(std::mem::take(literal)));
    }
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn program(mut self) -> Result<Program, String> {
        let mut items: Vec<Item> = Vec::new();
        let mut connector = Connector::First;
        loop {
            if self.tokens.get(self.index).is_none() {
                if matches!(connector, Connector::And | Connector::Or) {
                    return Err(format!(
                        "unexpected end of line after '{}'",
                        connector.text()
                    ));
                }
                break;
            }
            let pipeline = self.pipeline()?;
            // A terminal `&` makes this pipeline asynchronous and, like `;`,
            // lets the next item run unconditionally.
            let background = matches!(self.tokens.get(self.index), Some(Token::Op(Op::Ampersand)));
            if background {
                self.index += 1;
            }
            items.push(Item {
                connector,
                pipeline,
                background,
            });
            if background {
                // `&` is its own separator: the next item runs unconditionally.
                connector = Connector::Sequence;
                continue;
            }
            match self.tokens.get(self.index) {
                None => break,
                Some(Token::Op(Op::Semicolon)) => {
                    self.index += 1;
                    connector = Connector::Sequence;
                }
                Some(Token::Op(Op::And)) => {
                    self.index += 1;
                    connector = Connector::And;
                }
                Some(Token::Op(Op::Or)) => {
                    self.index += 1;
                    connector = Connector::Or;
                }
                Some(Token::Op(op)) => return Err(format!("unexpected '{}'", op.text())),
                Some(Token::Word(_)) => return Err("unexpected word after a command".into()),
            }
        }
        Ok(Program { items })
    }

    fn pipeline(&mut self) -> Result<Pipeline, String> {
        let mut commands = vec![self.command()?];
        while matches!(self.tokens.get(self.index), Some(Token::Op(Op::Pipe))) {
            self.index += 1;
            if self.tokens.get(self.index).is_none() {
                return Err("unexpected end of line after '|'".into());
            }
            commands.push(self.command()?);
        }
        Ok(Pipeline { commands })
    }

    fn command(&mut self) -> Result<Command, String> {
        let mut words = Vec::new();
        let mut redirects = Vec::new();
        loop {
            match self.tokens.get(self.index).cloned() {
                Some(Token::Word(word)) => {
                    self.index += 1;
                    words.push(word);
                }
                Some(Token::Op(op)) => {
                    let Some(kind) = redirect_kind(op) else {
                        break;
                    };
                    self.index += 1;
                    match self.tokens.get(self.index).cloned() {
                        Some(Token::Word(target)) => {
                            self.index += 1;
                            redirects.push(Redirect { kind, target });
                        }
                        _ => {
                            return Err(format!("expected a file name after '{}'", op.text()));
                        }
                    }
                }
                None => break,
            }
        }
        if words.is_empty() && redirects.is_empty() {
            return Err(match self.tokens.get(self.index) {
                Some(Token::Op(op)) => format!("unexpected '{}'", op.text()),
                _ => "unexpected end of line".into(),
            });
        }
        Ok(Command { words, redirects })
    }
}

fn redirect_kind(op: Op) -> Option<RedirectKind> {
    match op {
        Op::Write => Some(RedirectKind::Write),
        Op::Append => Some(RedirectKind::Append),
        Op::Read => Some(RedirectKind::Read),
        Op::Semicolon | Op::And | Op::Or | Op::Ampersand | Op::Pipe => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<Word> {
        let program = parse(line).expect("line parses");
        let mut words = Vec::new();
        for item in program.items {
            for command in item.pipeline.commands {
                words.extend(command.words);
            }
        }
        words
    }

    fn parts(line: &str) -> Vec<Vec<Part>> {
        words(line)
            .into_iter()
            .map(|word| word.parts().to_vec())
            .collect()
    }

    fn literal(text: &str) -> Part {
        Part::Literal(text.to_string())
    }

    #[test]
    fn splits_words_on_blanks_and_punctuation() {
        assert_eq!(
            parts("echo one two"),
            vec![
                vec![literal("echo")],
                vec![literal("one")],
                vec![literal("two")]
            ]
        );
    }

    #[test]
    fn single_quotes_keep_everything_literal() {
        assert_eq!(
            parts("echo 'a $b  # c'"),
            vec![vec![literal("echo")], vec![literal("a $b  # c")]]
        );
        assert_eq!(
            parts("echo ''"),
            vec![vec![literal("echo")], vec![literal("")]]
        );
    }

    #[test]
    fn double_quotes_expand_without_splitting() {
        assert_eq!(
            parts("echo \"a $b c\""),
            vec![
                vec![literal("echo")],
                vec![
                    literal("a "),
                    Part::Quoted(Expansion::Variable("b".into())),
                    literal(" c"),
                ]
            ]
        );
        assert_eq!(
            parts("echo \"\""),
            vec![vec![literal("echo")], vec![literal("")]]
        );
    }

    #[test]
    fn backslash_escapes_and_double_quote_rules() {
        assert_eq!(
            parts(r"echo a\ b"),
            vec![vec![literal("echo")], vec![literal("a b")]]
        );
        assert_eq!(
            parts(r#"echo "a\"b""#),
            vec![vec![literal("echo")], vec![literal("a\"b")]]
        );
        // A backslash before an ordinary character stays in double quotes.
        assert_eq!(
            parts(r#"echo "a\n""#),
            vec![vec![literal("echo")], vec![literal("a\\n")]]
        );
    }

    #[test]
    fn variable_and_status_expansions() {
        assert_eq!(
            parts("echo $HOME"),
            vec![
                vec![literal("echo")],
                vec![Part::Unquoted(Expansion::Variable("HOME".into()))]
            ]
        );
        assert_eq!(
            parts("echo ${_x9}"),
            vec![
                vec![literal("echo")],
                vec![Part::Unquoted(Expansion::Variable("_x9".into()))]
            ]
        );
        assert_eq!(
            parts("echo $?"),
            vec![
                vec![literal("echo")],
                vec![Part::Unquoted(Expansion::Status)]
            ]
        );
        assert_eq!(
            parts("echo \"$?{}\""),
            vec![
                vec![literal("echo")],
                vec![Part::Quoted(Expansion::Status), literal("{}")]
            ]
        );
    }

    #[test]
    fn a_dollar_without_a_name_is_literal() {
        assert_eq!(
            parts("echo $ $1 $@"),
            vec![
                vec![literal("echo")],
                vec![literal("$")],
                vec![literal("$1")],
                vec![literal("$@")],
            ]
        );
    }

    #[test]
    fn comments_start_only_at_a_token_boundary() {
        assert_eq!(
            parts("echo a#b # tail"),
            vec![vec![literal("echo")], vec![literal("a#b")]]
        );
        assert_eq!(parse("# only a comment").unwrap().items, vec![]);
        // `#` inside quotes is ordinary text.
        assert_eq!(
            parts("echo '#'"),
            vec![vec![literal("echo")], vec![literal("#")]]
        );
    }

    #[test]
    fn tilde_is_flagged_only_at_word_start() {
        let parsed = words("echo ~ ~/src ~user a~b");
        assert!(parsed[1].tilde());
        assert!(parsed[1].parts().is_empty());
        assert!(parsed[2].tilde());
        assert_eq!(parsed[2].parts(), &[literal("/src")]);
        assert!(!parsed[3].tilde());
        assert_eq!(parsed[3].parts(), &[literal("~user")]);
        assert!(!parsed[4].tilde());
        assert_eq!(parsed[4].parts(), &[literal("a~b")]);
    }

    #[test]
    fn separators_build_items_with_connectors() {
        let program = parse("a && b || c ; d").unwrap();
        let connectors: Vec<Connector> = program.items.iter().map(|item| item.connector).collect();
        assert_eq!(
            connectors,
            vec![
                Connector::First,
                Connector::And,
                Connector::Or,
                Connector::Sequence
            ]
        );
        assert_eq!(program.items.len(), 4);
    }

    #[test]
    fn pipelines_keep_every_command() {
        let program = parse("a | b | c").unwrap();
        assert_eq!(program.items.len(), 1);
        assert_eq!(program.items[0].pipeline.commands.len(), 3);
    }

    #[test]
    fn redirections_record_their_kind() {
        let program = parse("echo hi > out >> log < in").unwrap();
        let redirects = &program.items[0].pipeline.commands[0].redirects;
        assert_eq!(redirects.len(), 3);
        assert_eq!(redirects[0].kind, RedirectKind::Write);
        assert_eq!(redirects[1].kind, RedirectKind::Append);
        assert_eq!(redirects[2].kind, RedirectKind::Read);
        assert_eq!(redirects[0].target.parts(), &[literal("out")]);
    }

    #[test]
    fn a_redirect_only_command_is_valid() {
        let program = parse("> out").unwrap();
        assert!(program.items[0].pipeline.commands[0].words.is_empty());
        assert_eq!(program.items[0].pipeline.commands[0].redirects.len(), 1);
    }

    #[test]
    fn trailing_separators_are_accepted() {
        assert_eq!(parse("echo hi;").unwrap().items.len(), 1);
        assert_eq!(parse("echo hi ; ").unwrap().items.len(), 1);
    }

    #[test]
    fn unbalanced_and_missing_pieces_are_errors() {
        for (line, expected) in [
            ("echo 'a", "unclosed single quote"),
            ("echo \"a", "unclosed double quote"),
            ("echo ${a", "unclosed '${'"),
            ("echo ${a-b}", "bad substitution"),
            ("echo a\\", "trailing backslash"),
            ("| echo", "unexpected '|'"),
            ("; echo", "unexpected ';'"),
            ("echo a &&", "unexpected end of line after '&&'"),
            ("echo a ||", "unexpected end of line after '||'"),
            ("echo a |", "unexpected end of line after '|'"),
            ("echo a ; ; b", "unexpected ';'"),
            ("echo >", "expected a file name after '>'"),
            ("echo < >", "expected a file name after '<'"),
            ("cat <<EOF", "here-documents are not implemented yet"),
        ] {
            let error = parse(line).expect_err(line);
            assert!(
                error.contains(expected),
                "line {line:?} gave {error:?}, wanted {expected:?}"
            );
        }
    }

    #[test]
    fn ampersand_marks_the_preceding_pipeline_background() {
        let program = parse("sleep 1 &").unwrap();
        assert_eq!(program.items.len(), 1);
        assert!(program.items[0].background);

        // `&` separates items; the next item runs unconditionally.
        let program = parse("sleep 1 & echo after").unwrap();
        assert_eq!(program.items.len(), 2);
        assert!(program.items[0].background);
        assert!(!program.items[1].background);
        assert_eq!(program.items[1].connector, Connector::Sequence);

        // A whole pipeline is one background job.
        let program = parse("cat log | grep -v noise &").unwrap();
        assert!(program.items[0].background);
        assert_eq!(program.items[0].pipeline.commands.len(), 2);

        // A trailing `&` after a separator is fine.
        assert_eq!(parse("echo a; echo b &").unwrap().items.len(), 2);
        assert!(parse("&& echo").is_err());
        assert!(parse("| echo").is_err(), "a pipe still needs a left side");
    }

    #[test]
    fn a_lone_ampersand_is_an_error_and_double_ampersand_is_and() {
        assert_eq!(parse("a && b").unwrap().items.len(), 2);
        for line in ["& echo", "echo a & & echo b", "&"] {
            let error = parse(line).expect_err(line);
            assert!(
                error.contains("unexpected '&'"),
                "line {line:?} gave {error:?}"
            );
        }
    }

    #[test]
    fn identifiers_are_validated() {
        assert!(is_identifier("PATH"));
        assert!(is_identifier("_x9"));
        assert!(!is_identifier(""));
        assert!(!is_identifier("1x"));
        assert!(!is_identifier("a-b"));
    }
}
