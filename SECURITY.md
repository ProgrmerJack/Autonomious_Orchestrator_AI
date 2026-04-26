# Security Model

The orchestrator treats every worker, tool result, webpage, and document as untrusted input.

## Default Rules

- Workers must declare intended actions before they run.
- The supervisor routes and checkpoints work, but does not directly perform research actions.
- The policy engine blocks forbidden actions before adapter code is invoked.
- Durable memory accepts only compressed candidates with evidence and sufficient confidence.
- Host execution is disabled by default; sandbox providers return deterministic dry-run records unless explicitly configured.

## Forbidden by Default

- Persistent administrator credentials.
- System registry mutation.
- Security product or Defender policy modification.
- Credential file reads.
- Unscoped shell execution on the host.
- Network egress to undeclared hosts.

## Production Hardening Checklist

- Run browser and code execution workers inside c/ua, Firecracker, Kata, or another VM-backed sandbox.
- Route outbound traffic through a zero-trust egress proxy.
- Use short-lived scoped credentials injected outside the LLM context window.
- Persist run artifacts outside the sandbox only after verification.
- Keep the Rust body process small, audited, and separately permissioned from the Python brain.
