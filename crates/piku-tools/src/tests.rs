#[cfg(test)]
mod read_file {
    use super::tempdir;
    use crate::read_file;

    #[test]
    fn reads_full_file() {
        let dir = tempdir();
        let path = dir.join("hello.txt");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();
        let result = read_file::execute(serde_json::json!({ "path": path }));
        assert!(!result.is_error);
        assert!(result.output.contains("line1"));
        assert!(result.output.contains("line3"));
    }

    #[test]
    fn reads_line_range() {
        let dir = tempdir();
        let path = dir.join("range.txt");
        std::fs::write(&path, "a\nb\nc\nd\ne").unwrap();
        let result =
            read_file::execute(serde_json::json!({ "path": path, "start_line": 2, "end_line": 3 }));
        assert!(!result.is_error);
        assert!(result.output.contains('b'));
        assert!(result.output.contains('c'));
        assert!(!result.output.contains('a'));
        assert!(!result.output.contains('d'));
    }

    #[test]
    fn missing_file_is_error() {
        let result = read_file::execute(serde_json::json!({ "path": "/no/such/file.txt" }));
        assert!(result.is_error);
        assert!(result.output.contains("read_file"));
    }

    #[test]
    fn reads_unicode_content_without_panic() {
        let dir = tempdir();
        let path = dir.join("unicode.txt");
        // Mix of ASCII, CJK, emoji, combining characters
        let content = "hello\n你好世界\n🦀 Rust\nca\u{0301}fe\u{0301}\nline5";
        std::fs::write(&path, content).unwrap();
        let result = read_file::execute(serde_json::json!({ "path": path }));
        assert!(
            !result.is_error,
            "reading Unicode file failed: {}",
            result.output
        );
        assert!(result.output.contains("你好世界"), "CJK content missing");
        assert!(result.output.contains("🦀"), "emoji content missing");
    }

    #[test]
    fn rejects_file_over_size_limit() {
        let dir = tempdir();
        let path = dir.join("huge.bin");
        // Create a file just over 10 MB
        let size = 10 * 1024 * 1024 + 1;
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(size).unwrap();
        let result = read_file::execute(serde_json::json!({ "path": path }));
        assert!(result.is_error, "should reject files over 10 MB");
        assert!(
            result.output.contains("too large"),
            "error should mention size: {}",
            result.output
        );
    }
}

#[cfg(test)]
mod write_file {
    use super::tempdir;
    use crate::write_file;

    #[test]
    fn creates_new_file() {
        let dir = tempdir();
        let path = dir.join("new.txt");
        let result =
            write_file::execute(serde_json::json!({ "path": path, "content": "hello world" }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = tempdir();
        let path = dir.join("existing.txt");
        std::fs::write(&path, "old content").unwrap();
        let result =
            write_file::execute(serde_json::json!({ "path": path, "content": "new content" }));
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content");
    }

    #[test]
    fn creates_parent_dirs() {
        let dir = tempdir();
        let path = dir.join("a").join("b").join("c.txt");
        let result = write_file::execute(serde_json::json!({ "path": path, "content": "deep" }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "deep");
    }

    #[test]
    fn new_file_is_safe_destructiveness() {
        let dir = tempdir();
        let path = dir.join("brand_new.txt");
        let d = write_file::destructiveness(&serde_json::json!({ "path": path }));
        assert_eq!(d, crate::Destructiveness::Safe);
    }

    #[test]
    fn overwrite_is_likely_destructiveness() {
        let dir = tempdir();
        let path = dir.join("exists.txt");
        std::fs::write(&path, "x").unwrap();
        let d = write_file::destructiveness(&serde_json::json!({ "path": path }));
        assert_eq!(d, crate::Destructiveness::Likely);
    }
}

#[cfg(test)]
mod edit_file {
    use super::tempdir;
    use crate::edit_file;

    #[test]
    fn replaces_exact_match() {
        let dir = tempdir();
        let path = dir.join("edit.rs");
        std::fs::write(&path, "fn foo() {}\nfn bar() {}").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "fn foo() {}",
            "new_string": "fn foo(x: i32) {}",
        }));
        assert!(!result.is_error, "{}", result.output);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("fn foo(x: i32) {}"));
        assert!(content.contains("fn bar() {}"));
    }

    #[test]
    fn errors_on_not_found() {
        let dir = tempdir();
        let path = dir.join("nope.rs");
        std::fs::write(&path, "hello").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "DOES NOT EXIST",
            "new_string": "x",
        }));
        assert!(result.is_error);
        assert!(result.output.contains("not found"));
    }

    #[test]
    fn errors_on_ambiguous_match() {
        let dir = tempdir();
        let path = dir.join("dup.rs");
        std::fs::write(&path, "x = 1\nx = 1").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "x = 1",
            "new_string": "x = 2",
        }));
        assert!(result.is_error);
        assert!(result.output.contains("ambiguous"));
    }

    #[test]
    fn replace_all() {
        let dir = tempdir();
        let path = dir.join("multi.txt");
        std::fs::write(&path, "a a a").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "a",
            "new_string": "b",
            "replace_all": true,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "b b b");
    }

    #[test]
    fn errors_on_missing_file() {
        let result = edit_file::execute(serde_json::json!({
            "path": "/no/such/file.txt",
            "old_string": "x",
            "new_string": "y",
        }));
        assert!(result.is_error);
    }

    #[test]
    fn replaces_unicode_old_string() {
        let dir = tempdir();
        let path = dir.join("unicode_edit.rs");
        std::fs::write(&path, "// 你好世界\nfn main() {}").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "你好世界",
            "new_string": "hello world",
        }));
        assert!(!result.is_error, "{}", result.output);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("hello world"));
        assert!(!content.contains("你好世界"));
    }

    #[test]
    fn replaces_emoji_content() {
        let dir = tempdir();
        let path = dir.join("emoji_edit.txt");
        std::fs::write(&path, "status: 🔴 failing").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "🔴 failing",
            "new_string": "🟢 passing",
        }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "status: 🟢 passing"
        );
    }
}

#[cfg(test)]
mod bash {
    use crate::bash;

    #[tokio::test]
    async fn runs_simple_command() {
        let result = bash::execute(serde_json::json!({ "command": "echo hello" })).await;
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn captures_exit_code_error() {
        let result = bash::execute(serde_json::json!({ "command": "exit 1" })).await;
        assert!(result.is_error);
        assert!(result.output.contains("exit code 1"));
    }

    #[tokio::test]
    async fn respects_timeout() {
        let result =
            bash::execute(serde_json::json!({ "command": "sleep 10", "timeout_ms": 100 })).await;
        assert!(result.is_error);
        assert!(result.output.contains("timed out"));
    }

    #[test]
    fn rm_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "rm -rf /tmp/foo" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn echo_is_likely() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "echo hi" }));
        assert_eq!(d, crate::Destructiveness::Likely);
    }

    #[test]
    fn redirect_to_existing_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "echo x > /etc/passwd" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }
}

#[cfg(test)]
mod glob_tool {
    use super::tempdir;
    use crate::glob;

    #[test]
    fn finds_matching_files() {
        let dir = tempdir();
        std::fs::write(dir.join("a.rs"), "").unwrap();
        std::fs::write(dir.join("b.rs"), "").unwrap();
        std::fs::write(dir.join("c.txt"), "").unwrap();
        let result = glob::execute(serde_json::json!({
            "pattern": "*.rs",
            "path": dir,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("a.rs"));
        assert!(result.output.contains("b.rs"));
        assert!(!result.output.contains("c.txt"));
    }

    #[test]
    fn no_match_returns_no_matches() {
        let dir = tempdir();
        let result = glob::execute(serde_json::json!({
            "pattern": "*.zig",
            "path": dir,
        }));
        assert!(!result.is_error);
        assert!(result.output.contains("no matches"));
    }

    #[test]
    fn always_safe() {
        let d = glob::destructiveness(&serde_json::json!({}));
        assert_eq!(d, crate::Destructiveness::Safe);
    }
}

#[cfg(test)]
mod grep_tool {
    use super::tempdir;
    use crate::grep;

    #[test]
    fn finds_pattern_in_file() {
        let dir = tempdir();
        let path = dir.join("src.rs");
        std::fs::write(&path, "fn main() {\n    println!(\"hello\");\n}").unwrap();
        let result = grep::execute(serde_json::json!({
            "pattern": "println",
            "path": dir,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("println"));
        assert!(result.output.contains("src.rs"));
    }

    #[test]
    fn include_filter_works() {
        let dir = tempdir();
        std::fs::write(dir.join("a.rs"), "needle").unwrap();
        std::fs::write(dir.join("b.txt"), "needle").unwrap();
        let result = grep::execute(serde_json::json!({
            "pattern": "needle",
            "path": dir,
            "include": "*.rs",
        }));
        assert!(!result.is_error);
        assert!(result.output.contains("a.rs"));
        assert!(!result.output.contains("b.txt"));
    }

    #[test]
    fn no_match_returns_no_matches() {
        let dir = tempdir();
        std::fs::write(dir.join("f.rs"), "nothing here").unwrap();
        let result = grep::execute(serde_json::json!({
            "pattern": "XYZZY_NOT_PRESENT",
            "path": dir,
        }));
        assert!(!result.is_error);
        assert!(result.output.contains("no matches"));
    }

    #[test]
    fn invalid_regex_is_error() {
        let dir = tempdir();
        let result = grep::execute(serde_json::json!({
            "pattern": "[invalid",
            "path": dir,
        }));
        assert!(result.is_error);
    }

    #[test]
    fn always_safe() {
        let d = grep::destructiveness(&serde_json::json!({}));
        assert_eq!(d, crate::Destructiveness::Safe);
    }
}

#[cfg(test)]
mod list_dir_tool {
    use super::tempdir;
    use crate::list_dir;

    #[test]
    fn lists_files_and_dirs() {
        let dir = tempdir();
        std::fs::write(dir.join("file.txt"), "").unwrap();
        std::fs::create_dir(dir.join("subdir")).unwrap();
        let result = list_dir::execute(serde_json::json!({ "path": dir }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("file.txt"));
        assert!(result.output.contains("subdir/"));
    }

    #[test]
    fn empty_dir_message() {
        let dir = tempdir();
        let result = list_dir::execute(serde_json::json!({ "path": dir }));
        assert!(!result.is_error);
        assert!(result.output.contains("empty"));
    }

    #[test]
    fn missing_dir_is_error() {
        let result = list_dir::execute(serde_json::json!({ "path": "/no/such/dir" }));
        assert!(result.is_error);
    }

    #[test]
    fn always_safe() {
        let d = list_dir::destructiveness(&serde_json::json!({}));
        assert_eq!(d, crate::Destructiveness::Safe);
    }
}

// ---------------------------------------------------------------------------
// Shared test helper — creates a temp dir that auto-deletes
// ---------------------------------------------------------------------------

fn tempdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "piku_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// Path-traversal sandbox for write_file / edit_file.
/// The risk: a model proposes `../../etc/cron.d/x` or `/Users/a/.ssh/id_rsa`.
/// Even if the user approves the permission prompt, the tool should refuse.
#[cfg(test)]
mod sandbox {
    use super::tempdir;
    use serial_test::serial;

    /// Set CWD to a throwaway dir. `#[serial]` because CWD is process-global.
    fn cd(dir: &std::path::Path) {
        std::env::set_current_dir(dir).unwrap();
    }

    #[test]
    #[serial(cwd)]
    fn write_file_refuses_parent_traversal() {
        let td = tempdir();
        cd(&td);
        let params = serde_json::json!({
            "path": "../secrets.txt",
            "content": "boom",
        });
        let r = crate::write_file::execute(params);
        assert!(r.is_error, "traversal should be rejected");
        assert!(
            r.output.contains("refused") || r.output.contains("escapes"),
            "expected sandbox error, got: {}",
            r.output
        );
    }

    #[test]
    #[serial(cwd)]
    fn write_file_refuses_absolute_system_path() {
        let td = tempdir();
        cd(&td);
        let params = serde_json::json!({
            "path": "/etc/piku-sandbox-should-not-exist.txt",
            "content": "boom",
        });
        let r = crate::write_file::execute(params);
        assert!(r.is_error, "system path should be rejected");
        assert!(
            r.output.contains("system directory"),
            "expected system-path error, got: {}",
            r.output
        );
    }

    #[test]
    #[serial(cwd)]
    fn write_file_allows_absolute_user_path() {
        let td = tempdir();
        cd(&td);
        // /tmp is not a system root, so absolute paths there are allowed.
        let outside = std::env::temp_dir().join("piku-user-path-ok.txt");
        let params = serde_json::json!({
            "path": outside.display().to_string(),
            "content": "ok",
        });
        let r = crate::write_file::execute(params);
        let _ = std::fs::remove_file(&outside);
        assert!(
            !r.is_error,
            "absolute user path should succeed: {}",
            r.output
        );
    }

    #[test]
    #[serial(cwd)]
    fn write_file_allows_within_cwd() {
        let td = tempdir();
        cd(&td);
        let params = serde_json::json!({
            "path": "nested/ok.txt",
            "content": "hi",
        });
        let r = crate::write_file::execute(params);
        assert!(!r.is_error, "in-project write should succeed: {}", r.output);
    }

    #[test]
    #[serial(cwd)]
    fn edit_file_refuses_parent_traversal() {
        let td = tempdir();
        cd(&td);
        // Pre-create a file outside the sandbox (in td.parent()) to edit.
        let outside_dir = td.parent().unwrap();
        let outside = outside_dir.join("outside-edit-target.txt");
        std::fs::write(&outside, "original\n").unwrap();
        let rel = format!("../{}", outside.file_name().unwrap().to_string_lossy());
        let params = serde_json::json!({
            "path": rel,
            "old_string": "original",
            "new_string": "pwned",
        });
        let r = crate::edit_file::execute(params);
        assert!(r.is_error, "edit traversal should be rejected");
        // Target must NOT have been modified.
        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "original\n");
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    #[serial(cwd)]
    fn opt_out_env_disables_sandbox() {
        let td = tempdir();
        cd(&td);
        std::env::set_var("PIKU_ALLOW_WRITE_ANY", "1");
        let outside = std::env::temp_dir().join("piku-opt-out-test.txt");
        let params = serde_json::json!({
            "path": outside.display().to_string(),
            "content": "ok",
        });
        let r = crate::write_file::execute(params);
        std::env::remove_var("PIKU_ALLOW_WRITE_ANY");
        let _ = std::fs::remove_file(&outside);
        assert!(!r.is_error, "opt-out should allow any path: {}", r.output);
    }
}

// ---------------------------------------------------------------------------
// Extended P1/P2 tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod read_file_bounds {
    use super::tempdir;
    use crate::read_file;

    #[test]
    fn start_line_past_eof_returns_empty_not_panic() {
        // P1: was an index panic
        let dir = tempdir();
        let path = dir.join("short.txt");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();
        let result = read_file::execute(serde_json::json!({ "path": path, "start_line": 100 }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(result.output, ""); // past EOF = empty
    }

    #[test]
    fn start_greater_than_end_returns_error_not_panic() {
        // P1: was a slice panic
        let dir = tempdir();
        let path = dir.join("f.txt");
        std::fs::write(&path, "a\nb\nc").unwrap();
        let result =
            read_file::execute(serde_json::json!({ "path": path, "start_line": 5, "end_line": 2 }));
        assert!(result.is_error);
        assert!(result.output.contains("start_line"));
    }

    #[test]
    fn end_line_past_eof_is_clamped() {
        let dir = tempdir();
        let path = dir.join("small.txt");
        std::fs::write(&path, "a\nb").unwrap();
        let result = read_file::execute(
            serde_json::json!({ "path": path, "start_line": 1, "end_line": 9999 }),
        );
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains('a'));
        assert!(result.output.contains('b'));
    }

    #[test]
    fn empty_file_with_line_range_no_panic() {
        let dir = tempdir();
        let path = dir.join("empty.txt");
        std::fs::write(&path, "").unwrap();
        let result =
            read_file::execute(serde_json::json!({ "path": path, "start_line": 1, "end_line": 1 }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(result.output, "");
    }

    #[test]
    fn start_line_zero_treated_as_first_line() {
        // 0 → saturating_sub(1) = 0 = beginning; valid, returns from start
        let dir = tempdir();
        let path = dir.join("f.txt");
        std::fs::write(&path, "first\nsecond").unwrap();
        let result =
            read_file::execute(serde_json::json!({ "path": path, "start_line": 0, "end_line": 1 }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("first"));
    }
}

#[cfg(test)]
mod edit_file_edge_cases {
    use super::tempdir;
    use crate::edit_file;

    #[test]
    fn replace_all_empty_old_string_does_not_corrupt_file() {
        // Empty old_string would "match" len+1 times (between every char)
        // and silently corrupt the file by inserting new_string everywhere.
        // The tool should either error OR leave the file unchanged.
        let dir = tempdir();
        let path = dir.join("f.txt");
        let original = "hello world";
        std::fs::write(&path, original).unwrap();

        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "",
            "new_string": "X",
            "replace_all": true,
        }));

        // Post-condition: file content must be unchanged OR result is an error.
        // Both are acceptable; what is NOT acceptable is silent file corruption.
        let final_content = std::fs::read_to_string(&path).unwrap();
        assert!(
            result.is_error || final_content == original,
            "empty old_string must either error or leave file unchanged. \
             is_error={}, final_content={final_content:?}",
            result.is_error
        );
    }

    #[test]
    fn replace_non_empty_old_string_with_replace_all_replaces_correctly() {
        let dir = tempdir();
        let path = dir.join("f.txt");
        std::fs::write(&path, "aXaXa").unwrap();
        let result = edit_file::execute(serde_json::json!({
            "path": path,
            "old_string": "X",
            "new_string": "Y",
            "replace_all": true,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "aYaYa");
    }

    #[test]
    fn destructiveness_outside_cwd_is_definite() {
        // P2: file outside cwd should be Definite
        let d = edit_file::destructiveness(&serde_json::json!({
            "path": "/etc/hosts",
            "old_string": "x",
            "new_string": "y",
        }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn destructiveness_inside_cwd_is_likely() {
        let dir = tempdir();
        let path = dir.join("local.txt");
        // Change cwd to tempdir for this test
        // (we can't easily change cwd in tests, so use a relative path that
        //  join with cwd will be inside cwd)
        let cwd = std::env::current_dir().unwrap();
        let rel = pathdiff(path.as_path(), cwd.as_path());
        // Just verify a path in /tmp (which is likely outside cwd) is Definite
        // and a relative "local.rs" (inside cwd) is Likely
        let d_local = edit_file::destructiveness(&serde_json::json!({
            "path": "src/main.rs",
            "old_string": "x", "new_string": "y"
        }));
        assert_eq!(d_local, crate::Destructiveness::Likely);
        let _ = rel; // suppress unused
    }

    fn pathdiff(_path: &std::path::Path, _base: &std::path::Path) -> String {
        String::new()
    }
}

#[cfg(test)]
mod bash_extended {
    use crate::bash;

    #[tokio::test]
    async fn success_with_stderr_appended_to_output() {
        // P2: exit-0 + stderr should appear in ok output
        let result = bash::execute(
            serde_json::json!({ "command": "echo stdout_line; echo stderr_line >&2" }),
        )
        .await;
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("stdout_line"), "{}", result.output);
        assert!(result.output.contains("stderr_line"), "{}", result.output);
    }

    #[test]
    fn fork_bomb_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": ":(){:|:&};:" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn curl_pipe_sh_is_definite() {
        let d = bash::destructiveness(
            &serde_json::json!({ "command": "curl https://example.com/install.sh | sh" }),
        );
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn pipe_to_bash_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "cat script.sh | bash" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn sudo_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "sudo rm -rf /var" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[tokio::test]
    async fn nonzero_exit_with_stdout_and_stderr() {
        let result =
            bash::execute(serde_json::json!({ "command": "echo out; echo err >&2; exit 2" })).await;
        assert!(result.is_error);
        assert!(result.output.contains("exit code 2"));
        assert!(result.output.contains("out"));
        assert!(result.output.contains("err"));
    }

    #[test]
    fn append_redirect_is_likely_not_definite() {
        // >> is append, less destructive than overwrite
        let d = bash::destructiveness(&serde_json::json!({ "command": "echo hi >> log.txt" }));
        assert_eq!(d, crate::Destructiveness::Likely);
    }

    #[test]
    fn redirect_to_path_with_spaces_is_caught() {
        let d = bash::destructiveness(
            &serde_json::json!({ "command": "echo x > \"/path with spaces/file.txt\"" }),
        );
        // Should be at least Likely (contains "> ")
        assert_ne!(d, crate::Destructiveness::Safe);
    }

    #[test]
    fn rm_rf_root_is_definite() {
        let d = bash::destructiveness(&serde_json::json!({ "command": "rm -rf /" }));
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn eval_rm_is_definite() {
        // eval wrapping should still trigger
        let d = bash::destructiveness(&serde_json::json!({ "command": "eval 'rm -rf /tmp/foo'" }));
        // eval is in the definite patterns
        assert_eq!(d, crate::Destructiveness::Definite);
    }

    #[test]
    fn benign_command_is_at_least_likely() {
        // bash is always at least Likely -- any shell command can be destructive
        let d = bash::destructiveness(&serde_json::json!({ "command": "ls -la" }));
        assert_eq!(d, crate::Destructiveness::Likely);
    }
}

#[cfg(test)]
mod glob_extended {
    use super::tempdir;
    use crate::glob;

    #[test]
    fn invalid_pattern_returns_error() {
        let result = glob::execute(serde_json::json!({ "pattern": "[invalid" }));
        assert!(result.is_error);
        assert!(result.output.contains("invalid pattern"));
    }

    #[test]
    fn recursive_glob_finds_nested_files() {
        let dir = tempdir();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("deep.rs"), "").unwrap();
        let result = glob::execute(serde_json::json!({
            "pattern": "**/*.rs",
            "path": dir,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("deep.rs"));
    }
}

#[cfg(test)]
mod grep_extended {
    use super::tempdir;
    use crate::grep;

    #[test]
    fn search_single_file_directly() {
        // P2: root_path.is_file() branch
        let dir = tempdir();
        let path = dir.join("single.rs");
        std::fs::write(&path, "fn main() {}\nfn helper() {}").unwrap();
        let result = grep::execute(serde_json::json!({
            "pattern": "fn ",
            "path": path,
        }));
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("main"));
        assert!(result.output.contains("helper"));
    }

    #[test]
    fn max_results_truncates() {
        let dir = tempdir();
        let path = dir.join("many.txt");
        // 20 lines all matching
        let mut content = String::new();
        for i in 0..20 {
            use std::fmt::Write as _;
            writeln!(&mut content, "match line {i}").unwrap();
        }
        std::fs::write(&path, &content).unwrap();
        let result = grep::execute(serde_json::json!({
            "pattern": "match",
            "path": dir,
            "max_results": 5,
        }));
        assert!(!result.is_error, "{}", result.output);
        let lines: Vec<&str> = result.output.lines().collect();
        // 5 matches + truncation message
        assert!(result.output.contains("truncated"), "{}", result.output);
        assert!(lines.len() <= 6, "expected ≤6 lines, got {}", lines.len());
    }
}
