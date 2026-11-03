//! End-to-end lifecycle checks: the graceful-shutdown deadline must fire
//! when an upstream hangs a request hostage, and `oxllm status` must exit
//! non-zero when no daemon is running.

use std::io::Write;
use std::net::TcpListener;
use std::process::Command;
use std::time::{Duration, Instant};

/// The per-platform service unit path under a (throwaway) HOME.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn install_unit_path(home: &std::path::Path) -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    {
        home.join("Library/LaunchAgents/org.oxllm.proxy.plist")
    }
    #[cfg(target_os = "linux")]
    {
        home.join(".config/systemd/user/oxllm.service")
    }
}

fn oxllm() -> Command {
    Command::new(env!("CARGO_BIN_EXE_oxllm"))
}

/// A loopback port with (almost certainly) nothing listening on it.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// An upstream that accepts connections and never responds, so proxy
/// requests hang until their upstream timeout.
fn spawn_silent_upstream() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            // Hold every connection open, say nothing.
            drop(stream);
        }
    });
    port
}

/// `oxllm status` must exit non-zero when no daemon is on the port.
#[test]
fn status_exits_nonzero_when_no_daemon_listens() {
    let out = oxllm()
        .args(["status", "-p", &free_port().to_string()])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "status must fail when no daemon is running"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("not running"), "got: {}", stdout);
}
/// `oxllm install` must refuse to replace an existing service without
/// --force (issue #2 acceptance: never silently overwrite). Runs against a
/// throwaway HOME so the test is hermetic: it never touches (or depends
/// on) a real installation, and parallel runs cannot race on the shared
/// unit path.
#[test]
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn install_refuses_existing_unit_without_force() {
    let home = std::env::temp_dir().join(format!(
        "oxllm-install-test-home-{}-existing",
        std::process::id()
    ));
    let unit = install_unit_path(&home);

    std::fs::create_dir_all(unit.parent().unwrap()).unwrap();
    std::fs::write(&unit, "pre-existing installation").unwrap();

    let out = oxllm().arg("install").env("HOME", &home).output().unwrap();
    assert!(
        !out.status.success(),
        "install must refuse an existing unit without --force"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("already installed"), "got: {}", stdout);
    assert_eq!(
        std::fs::read_to_string(&unit).unwrap(),
        "pre-existing installation",
        "refusal must not modify the unit"
    );

    let _ = std::fs::remove_dir_all(&home);
}
/// `oxllm install` must refuse an occupied port before writing anything
/// (issue #8): a fresh install into a port conflict would produce a
/// crash-looping service. Runs against a throwaway HOME with no existing
/// installation, so the fresh-install preflight path is exercised without
/// touching (or depending on) a real installation.
#[test]
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn install_refuses_occupied_port() {
    let home = std::env::temp_dir().join(format!(
        "oxllm-install-test-home-{}-port",
        std::process::id()
    ));
    let unit = install_unit_path(&home);

    // Occupy a port and point a config at it.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = std::env::temp_dir().join(format!("oxllm-port-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.toml");
    std::fs::write(
        &config,
        format!(
            r#"
[server]
host = "127.0.0.1"
port = {port}
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "p"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "k"
models = ["m"]

[virtual_models]
vm = [{{ provider = "p", model = "m" }}]
"#
        ),
    )
    .unwrap();

    let out = oxllm()
        .args(["install", "--force", "--config", config.to_str().unwrap()])
        .env("HOME", &home)
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "install must refuse an occupied port"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("already in use"), "got: {}", stdout);
    assert!(!unit.exists(), "refusal must not write a unit file");

    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_dir(&dir);
    let _ = std::fs::remove_dir_all(&home);
}

/// The shutdown deadline must bound only the post-signal drain, never the
/// serve lifetime: a daemon with `shutdown_timeout_secs = 2` and no signal
/// must still be serving well past 2s. Regression test for the v0.1.19
/// bug where `tokio::time::timeout` wrapped the whole serve future and
/// KeepAlive restart-looped the daemon every (deadline + 1)s.
#[test]
fn serve_outlives_shutdown_deadline_without_signal() {
    let daemon_port = free_port();

    let dir = std::env::temp_dir().join(format!("oxllm-alive-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("daemon.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[server]
host = "127.0.0.1"
port = {daemon_port}
otel_endpoint = "http://127.0.0.1:4318"
shutdown_timeout_secs = 2

[[providers]]
name = "p"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "k"
models = ["m"]

[virtual_models]
vm = [{{ provider = "p", model = "m" }}]
"#
        ),
    )
    .unwrap();

    let mut child = oxllm()
        .args(["serve", "--config", config_path.to_str().unwrap()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();

    // Wait for the listener, then wait well past the 2s deadline.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "daemon never started listening");
        if TcpListener::bind(("127.0.0.1", daemon_port)).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(Duration::from_secs(6));

    assert!(
        child.try_wait().unwrap().is_none(),
        "daemon exited without a shutdown signal — the deadline is bounding the serve lifetime"
    );
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", daemon_port)).is_ok(),
        "daemon must still be accepting connections past the deadline"
    );

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_dir(&dir);
}

/// SIGTERM drains in-flight requests only until
/// `server.shutdown_timeout_secs` expires: a request stuck on a hung
/// upstream must not hold the shutdown hostage. With a 1s deadline and a
/// 30s upstream timeout, the process must exit within a few seconds of
/// SIGTERM (without the deadline it would stay up for the full drain).
#[test]
fn shutdown_deadline_exits_despite_hung_request() {
    let upstream_port = spawn_silent_upstream();
    let daemon_port = free_port();

    let dir = std::env::temp_dir().join(format!("oxllm-shutdown-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("daemon.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[server]
host = "127.0.0.1"
port = {daemon_port}
otel_endpoint = "http://127.0.0.1:4318"
upstream_timeout_secs = 30
shutdown_timeout_secs = 1

[[providers]]
name = "hung"
enabled = true
base_url = "http://127.0.0.1:{upstream_port}/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [{{ provider = "hung", model = "m" }}]
"#
        ),
    )
    .unwrap();

    let mut child = oxllm()
        .args(["serve", "--config", config_path.to_str().unwrap()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();

    // Wait for the listener to come up.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "daemon never started listening");
        if TcpListener::bind(("127.0.0.1", daemon_port)).is_err() {
            break; // port is taken => daemon is listening
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // A complete chat request whose upstream never responds: in-flight at
    // shutdown time.
    let body = r#"{"model":"vm","messages":[{"role":"user","content":"hi"}]}"#;
    let mut client = std::net::TcpStream::connect(("127.0.0.1", daemon_port)).unwrap();
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
    .unwrap();
    client.flush().unwrap();

    // SIGTERM, then the deadline must end the drain well before the 30s
    // upstream timeout would.
    let status = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success(), "failed to send SIGTERM");

    let started = Instant::now();
    let exit = loop {
        match child.try_wait().unwrap() {
            Some(exit) => break exit,
            None if started.elapsed() > Duration::from_secs(8) => {
                let _ = child.kill();
                panic!("daemon did not exit within 8s of SIGTERM; deadline not enforced");
            },
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "daemon took {:?} to exit; the 1s deadline did not bound the drain",
        started.elapsed()
    );
    assert_eq!(exit.code(), Some(0), "deadline exit should still be clean");

    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_dir(&dir);
}
