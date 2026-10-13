//! Per-launch browser admission and private, descriptor-relative handoff creation.
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};

use axum::http::{HeaderMap, HeaderName};
use nix::fcntl::{open, openat, OFlag};
use nix::sys::stat::{fstat, mkdirat, Mode};
use nix::unistd::geteuid;

pub(super) struct LaunchAuth {
    token: String,
    host: String,
    numeric_host: String,
    origin: String,
}

impl fmt::Debug for LaunchAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LaunchAuth")
            .field("origin", &self.origin)
            .field("token", &"[redacted]")
            .finish_non_exhaustive()
    }
}

impl LaunchAuth {
    pub(super) fn new(port: u16) -> anyhow::Result<Self> {
        let mut bytes = [0_u8; 32];
        getrandom::fill(&mut bytes).map_err(|_| anyhow::anyhow!("launch entropy unavailable"))?;
        let mut token = String::with_capacity(64);
        for byte in bytes {
            use std::fmt::Write as _;
            write!(&mut token, "{byte:02x}").expect("writing into String cannot fail");
        }
        Ok(Self {
            token,
            host: format!("localhost:{port}"),
            numeric_host: format!("127.0.0.1:{port}"),
            origin: format!("http://localhost:{port}"),
        })
    }

    pub(super) fn origin(&self) -> &str {
        &self.origin
    }

    pub(super) fn accepts_host(&self, host: &str) -> bool {
        host == self.host || host == self.numeric_host
    }

    pub(super) fn matches(&self, candidate: &str) -> bool {
        if candidate.len() != self.token.len() {
            return false;
        }
        candidate
            .bytes()
            .zip(self.token.bytes())
            .fold(0_u8, |diff, (a, b)| diff | (a ^ b))
            == 0
    }

    #[cfg(test)]
    pub(super) fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    pub(super) fn write_launch_file(&self, config_dir: &Path) -> anyhow::Result<PathBuf> {
        // Open each ancestor without following links, and retain directory handles
        // through creation. A pathname check followed by ordinary write would race.
        let absolute = if config_dir.is_absolute() {
            config_dir.to_path_buf()
        } else {
            std::env::current_dir()?.join(config_dir)
        };
        // Resolve an existing operator-selected root (including macOS /var and
        // /tmp aliases), then walk the resolved ancestors without following links.
        // Missing descendants are created only through those retained handles.
        let mut existing = absolute.as_path();
        let mut missing = Vec::new();
        while !existing.try_exists()? {
            missing.push(
                existing
                    .file_name()
                    .ok_or_else(|| io::Error::other("invalid launch directory"))?
                    .to_owned(),
            );
            existing = existing
                .parent()
                .ok_or_else(|| io::Error::other("invalid launch directory"))?;
        }
        let mut absolute = existing.canonicalize()?;
        for name in missing.iter().rev() {
            absolute.push(name);
        }
        let mut directory = open_root()?;
        for component in absolute.components() {
            match component {
                Component::RootDir => {}
                Component::Normal(name) => {
                    directory = open_child_dir(&directory, name.as_bytes(), false)?;
                }
                _ => anyhow::bail!("launch directory must have a normalized path"),
            }
        }
        directory = open_child_dir(&directory, b"_web", true)?;
        let launch_id = crate::new_session_id();
        directory = open_child_dir(&directory, launch_id.as_bytes(), true)?;
        let fd = openat(
            &directory,
            "launch.html",
            OFlag::O_WRONLY | OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
            Mode::from_bits_truncate(0o600),
        )?;
        let mut file = File::from(fd);
        write!(file, "<!doctype html><meta name=\"referrer\" content=\"no-referrer\"><meta http-equiv=\"refresh\" content=\"0;url={}/#token={}\"><title>Open Piku</title><a href=\"{}/#token={}\">Open Piku</a>", self.origin, self.token, self.origin, self.token)?;
        file.sync_all()?;
        Ok(absolute.join("_web").join(launch_id).join("launch.html"))
    }
}

pub(super) fn single_header(headers: &HeaderMap, name: HeaderName) -> Option<&str> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    Some(value)
}

fn open_root() -> io::Result<File> {
    let fd = open(
        "/",
        OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
        Mode::empty(),
    )?;
    Ok(File::from(fd))
}

fn open_child_dir(parent: &File, name: &[u8], private: bool) -> io::Result<File> {
    let flags = OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC;
    let fd = match openat(parent, name, flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(nix::errno::Errno::ENOENT) => {
            match mkdirat(parent, name, Mode::from_bits_truncate(0o700)) {
                Ok(()) | Err(nix::errno::Errno::EEXIST) => {}
                Err(error) => return Err(error.into()),
            }
            openat(parent, name, flags, Mode::empty())?
        }
        Err(error) => return Err(error.into()),
    };
    let directory = File::from(fd);
    let stat = fstat(&directory)?;
    let uid = geteuid().as_raw();
    let owner_ok = stat.st_uid == uid || (!private && stat.st_uid == 0);
    let mode_ok = if private {
        stat.st_mode & 0o777 == 0o700
    } else {
        stat.st_mode & 0o022 == 0 || stat.st_mode & 0o1000 != 0
    };
    if !owner_ok || !mode_ok {
        return Err(io::Error::other(
            "unsafe launch directory ownership or permissions",
        ));
    }
    Ok(directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    #[test]
    fn capability_is_fresh_listener_bound_and_redacted() {
        let first = LaunchAuth::new(34567).unwrap();
        let second = LaunchAuth::new(34567).unwrap();
        assert_eq!(first.token.len(), 64);
        assert!(first.matches(&first.token));
        assert!(!first.matches(&second.token));
        assert!(first.accepts_host("localhost:34567"));
        assert!(first.accepts_host("127.0.0.1:34567"));
        for host in [
            "localhost:34568",
            "localhost",
            "evil.example:34567",
            "localhost:34567,evil",
        ] {
            assert!(!first.accepts_host(host));
        }
        assert!(!format!("{first:?}").contains(&first.token));
    }

    #[test]
    fn handoff_is_private_and_rejects_links_and_writable_ancestors() {
        // Home ancestors are non-writable; temporary system directories need not be.
        let base = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("launch-test-{}", crate::new_session_id()));
        let auth = LaunchAuth::new(34567).unwrap();
        let file = auth.write_launch_file(&base).unwrap();
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(file.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        let html = std::fs::read_to_string(&file).unwrap();
        assert!(html.contains(&format!("/#token={}", auth.token)));
        assert!(!html.contains("?token="));
        let linked = base.join("linked");
        std::fs::create_dir(&linked).unwrap();
        symlink(file.parent().unwrap(), linked.join("_web")).unwrap();
        assert!(auth.write_launch_file(&linked).is_err());
        let alias = base.join("alias");
        symlink(&linked, &alias).unwrap();
        assert!(auth.write_launch_file(&alias).is_err());
        std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(auth.write_launch_file(&base).is_err());
        std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o1700)).unwrap();
        let sticky = tempfile::tempdir().unwrap();
        std::fs::set_permissions(sticky.path(), std::fs::Permissions::from_mode(0o1777)).unwrap();
        let sticky_file = auth
            .write_launch_file(&sticky.path().join("config"))
            .unwrap();
        assert_eq!(
            std::fs::metadata(sticky_file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::set_permissions(sticky.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(auth
            .write_launch_file(&sticky.path().join("config"))
            .is_err());
        std::fs::set_permissions(sticky.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::remove_dir_all(base).unwrap();
    }
}
