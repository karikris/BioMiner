# Command Surface Policy

Status: accepted

The maintained production entrypoint is `biominer run --taxon <name> --rank auto|family|genus|species`, with explicit storage and workstore backends for local or cloud execution.

Step-specific commands remain available when they are reusable package workflows: registry build/audit, Flickr metadata polling, vision detect/screen/score/ablate, evidence join, comment review, and evaluation. One-off scripts, root wrappers, and prototype commands are removed once their behavior is covered by maintained package commands and tests.

Migration notes should live in current workflow docs, not generated report files. Removed commands must either be absent from docs or listed in `docs/deprecated_removed_commands.md` when users need a historical mapping.
