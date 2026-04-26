# HEARTBEAT

enabled: true
interval_seconds: 300
max_background_turns: 3

## Background Duties

- Continue recoverable runs that are not blocked on approval.
- Summarize newly completed worker outputs.
- Run unsupervised evaluations over recent event history.
- Avoid OS actions unless a human has approved the pending ticket.
