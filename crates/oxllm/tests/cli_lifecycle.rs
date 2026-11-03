//! End-to-end lifecycle checks: the graceful-shutdown deadline must fire
//! when an upstream hangs a request hostage, and `oxllm status` must exit
//! non-zero when no daemon is running.

use std::io::Write;
use std::net::TcpListener;
use std::process::Command;
use std::time::{Duration, Instant};

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
