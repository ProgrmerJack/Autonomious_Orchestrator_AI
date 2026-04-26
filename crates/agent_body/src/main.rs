use std::env;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "--health") {
        println!(
            "{{\"status\":\"ok\",\"name\":\"agent_body\",\"version\":\"0.1.0\"}}"
        );
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--describe") {
        println!(
            "{{\"capabilities\":[\"event-bridge\",\"stdin-actions\"]}}"
        );
        return Ok(());
    }
    serve()
}

fn serve() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    writeln!(
        stdout,
        "{{\"type\":\"body.started\",\"status\":\"ready\"}}"
    )?;
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let payload = escape_json(trimmed);
        writeln!(
            stdout,
            "{{\"type\":\"action.received\",\"status\":\"queued\",\"payload\":\"{}\"}}",
            payload
        )?;
        stdout.flush()?;
    }
    Ok(())
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other => escaped.push(other),
        }
    }
    escaped
}
