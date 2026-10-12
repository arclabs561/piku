//! Serialized append allocation with incremental history validation.
use super::{
    now_ms, validate_existing, validate_record_segment, EventScope, EventScopeKind, RunEvent,
    RunEventEnvelope, RunRecorder, RUN_RECORD_SCHEMA_VERSION,
};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

struct AppendLock(File);

impl AppendLock {
    fn acquire(file: &File) -> io::Result<Self> {
        let lock = file.try_clone()?;
        lock.try_lock().map_err(|error| match error {
            std::fs::TryLockError::WouldBlock => io::Error::new(
                io::ErrorKind::WouldBlock,
                "run record is busy; retry append",
            ),
            std::fs::TryLockError::Error(error) => error,
        })?;
        Ok(Self(lock))
    }
}

impl Drop for AppendLock {
    fn drop(&mut self) {
        // Forked descriptors may outlive this handle, so close alone is not
        // sufficient to release the shared open-file-description lock.
        if let Err(error) = self.0.unlock() {
            eprintln!("[piku] run record unlock failed: {error}");
        }
    }
}

impl RunRecorder {
    pub fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)?;
        let lock = AppendLock::acquire(&file)?;
        // File locks already follow aliases. Resolve final-component symlinks
        // so artifact references use the target's directory and basename;
        // preserve ordinary caller paths for RunHandle's path contract.
        let path = if fs::symlink_metadata(path)?.file_type().is_symlink() {
            path.canonicalize()?
        } else {
            path.to_path_buf()
        };
        let artifact_dir = path.with_extension("artifacts");
        let artifact_relative_dir = artifact_dir
            .file_name()
            .map(PathBuf::from)
            .ok_or_else(|| io::Error::other("run record path has no file name"))?;
        let mut recorder = Self {
            path,
            artifact_dir,
            artifact_relative_dir,
            session_id: session_id.into(),
            next_sequence: 0,
            validated_bytes: 0,
            writer: file,
        };
        recorder.refresh_sequence()?;
        drop(lock);
        Ok(recorder)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, turn_id: impl Into<String>, mut event: RunEvent) -> io::Result<u64> {
        self.append_scoped(
            EventScope::Turn {
                turn_id: turn_id.into(),
            },
            &mut event,
        )
    }

    pub fn append_run(&mut self, mut event: RunEvent) -> io::Result<u64> {
        self.append_scoped(EventScope::Run, &mut event)
    }

    fn append_scoped(&mut self, scope: EventScope, event: &mut RunEvent) -> io::Result<u64> {
        let actual_scope = match scope {
            EventScope::Run => EventScopeKind::Run,
            EventScope::Turn { .. } => EventScopeKind::Turn,
        };
        if actual_scope != event.scope_kind() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "run event has incompatible scope",
            ));
        }
        let _lock = AppendLock::acquire(&self.writer)?;
        self.refresh_sequence()?;
        let sequence = self.next_sequence;
        let next = sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("run event sequence overflow"))?;
        self.materialize_large_content(sequence, event)?;
        let envelope = RunEventEnvelope {
            schema_version: RUN_RECORD_SCHEMA_VERSION,
            sequence,
            recorded_at_ms: now_ms()?,
            session_id: self.session_id.clone(),
            scope,
            event: event.clone(),
        };
        // Serialize before touching the stream; failed writes cannot leave a
        // buffered tail that would be appended again by a destructor.
        let mut line = serde_json::to_vec(&envelope).map_err(io::Error::other)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        self.writer.flush()?;
        self.validated_bytes = self.writer.metadata()?.len();
        self.next_sequence = next;
        Ok(sequence)
    }

    fn refresh_sequence(&mut self) -> io::Result<()> {
        let metadata = self.writer.metadata()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let current = fs::metadata(&self.path)?;
            if (metadata.dev(), metadata.ino()) != (current.dev(), current.ino()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "run record was replaced",
                ));
            }
        }
        let length = metadata.len();
        if length < self.validated_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "run record was truncated",
            ));
        }
        if length == self.validated_bytes {
            return Ok(());
        }
        self.writer.seek(SeekFrom::Start(self.validated_bytes))?;
        let mut suffix = String::new();
        self.writer.read_to_string(&mut suffix)?;
        if !suffix.ends_with('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "run record has an incomplete final line",
            ));
        }
        let events = suffix
            .lines()
            .map(|line| {
                serde_json::from_str(line).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid appended run event: {error}"),
                    )
                })
            })
            .collect::<io::Result<Vec<RunEventEnvelope>>>()?;
        if self.validated_bytes == 0 {
            validate_existing(&events, &self.session_id)?;
        } else {
            validate_record_segment(&events, self.next_sequence, Some(&self.session_id))?;
        }
        let count =
            u64::try_from(events.len()).map_err(|_| io::Error::other("too many run events"))?;
        self.next_sequence = self
            .next_sequence
            .checked_add(count)
            .ok_or_else(|| io::Error::other("run event sequence overflow"))?;
        self.validated_bytes = length;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warning() -> RunEvent {
        RunEvent::Warning {
            message: "fixture".into(),
        }
    }

    #[test]
    fn held_append_lock_fails_busy_without_writing_and_releases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut first = RunRecorder::open(&path, "session").unwrap();
        let mut second = RunRecorder::open(&path, "session").unwrap();
        let lock = AppendLock::acquire(&first.writer).unwrap();
        assert_eq!(
            second.append("turn", warning()).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
        drop(lock);
        assert_eq!(second.append("turn", warning()).unwrap(), 0);
        assert_eq!(first.append("turn", warning()).unwrap(), 1);
    }

    #[test]
    fn stale_recorder_rejects_bad_suffix_and_session_change() {
        for suffix in ["not json\n", "{", "\n"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("run.jsonl");
            let mut recorder = RunRecorder::open(&path, "session").unwrap();
            recorder.append("turn", warning()).unwrap();
            OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(suffix.as_bytes())
                .unwrap();
            let before = fs::read(&path).unwrap();
            assert!(recorder.append("turn", warning()).is_err());
            assert_eq!(fs::read(&path).unwrap(), before);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut first = RunRecorder::open(&path, "first").unwrap();
        let mut second = RunRecorder::open(&path, "second").unwrap();
        first.append("turn", warning()).unwrap();
        assert!(second.append("turn", warning()).is_err());
        assert_eq!(crate::run_record::read_run_record(&path).unwrap().len(), 1);
    }

    #[test]
    fn artifact_io_failure_does_not_acknowledge_or_consume_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut recorder = RunRecorder::open(&path, "session").unwrap();
        fs::write(path.with_extension("artifacts"), "not a directory").unwrap();
        let result = recorder.append(
            "turn",
            RunEvent::AssistantMessage {
                content: crate::run_record::ContentRef::Inline {
                    text: "x".repeat(super::super::RUN_INLINE_CONTENT_LIMIT_BYTES + 1),
                },
            },
        );
        assert!(result.is_err());
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
        assert_eq!(recorder.append("turn", warning()).unwrap(), 0);
        assert_eq!(crate::run_record::read_run_record(&path).unwrap().len(), 1);
    }

    #[test]
    fn stale_recorder_rejects_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut recorder = RunRecorder::open(&path, "session").unwrap();
        recorder.append("turn", warning()).unwrap();
        fs::write(&path, "").unwrap();
        assert!(recorder.append("turn", warning()).is_err());
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn stale_recorder_rejects_replaced_inode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.jsonl");
        let mut recorder = RunRecorder::open(&path, "session").unwrap();
        let replacement = dir.path().join("replacement");
        fs::write(&replacement, "").unwrap();
        fs::rename(replacement, &path).unwrap();
        assert!(recorder.append("turn", warning()).is_err());
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }
}
