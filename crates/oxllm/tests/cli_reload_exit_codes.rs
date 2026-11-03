//! End-to-end exit-code checks for `oxllm reload` / `oxllm stop` (issue #7):
//! a reload that cannot reach a daemon, or that the daemon rejects, must
//! fail the command with a non-zero exit code instead of exiting 0.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;

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

/// Serves exactly one HTTP response — a 400 carrying the reason a daemon
/// would return for a rejected config — so the CLI's failure handling can
/// be exercised end to end.
fn spawn_400_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf);
        let body = r#"{"error":{"message":"Config validation failed: Virtual model 'vm' targets undefined provider 'missing-prov'","type":"invalid_request_error","code":400}}"#;
        let response = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    port
}

/// Spawns and reaps a child to obtain a PID that is guaranteed dead.
fn dead_pid() -> u32 {
    let mut child = Command::new("sleep").arg("30").spawn().unwrap();
    let pid = child.id();
    child.kill().unwrap();
    child.wait().unwrap();
    pid
}

/// `oxllm reload` with no daemon on the port must exit non-zero and say so.
#[test]
fn reload_exits_nonzero_when_no_daemon_listens() {
    let out = oxllm()
        .args(["reload", "-p", &free_port().to_string()])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "reload must fail when no daemon is running"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("not running"), "got: {}", stdout);
}

/// `oxllm reload` must also exit non-zero when the daemon rejects the new
/// config, and print the reason the daemon returned.
#[test]
fn reload_exits_nonzero_when_daemon_rejects_config() {
    let port = spawn_400_server();
    let out = oxllm()
        .args(["reload", "-p", &port.to_string()])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "reload must fail when the daemon rejects the config"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Config validation failed"),
        "got: {}",
        stdout
    );
}

/// `oxllm stop` must exit non-zero when the target PID is already gone.
#[test]
fn stop_exits_nonzero_for_dead_pid() {
    let out = oxllm()
        .args(["stop", "--pid", &dead_pid().to_string()])
        .output()
        .unwrap();
    assert!(!out.status.success(), "stop must fail for a dead PID");
}
