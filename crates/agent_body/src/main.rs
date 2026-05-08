use std::io;

fn main() -> io::Result<()> {
    agent_body::run(std::env::args().collect())
}
