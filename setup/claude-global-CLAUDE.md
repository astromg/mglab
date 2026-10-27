## OCA/OCM Shared Knowledge Vault (Obsidian)

There is a shared team knowledge base for the OCM observatory ecosystem at
`__HOME__/work/src/knowledge-base` (git: `araucaria-project/knowledge-base`, private) —
collective memory across OCM-related projects: architecture, infrastructure,
NATS subjects, domain knowledge, conventions, per-project context not in the code.

**Scope:** applies whenever the work relates to the OCM observatory — OCA, CAMK,
Akond, serverish, ocabox family, halina, TOI, ocadb, pyaraucaria, FITS/photometry
pipelines, etc. It does NOT apply to unrelated work such as CUBES AG; there,
neither read nor write this vault.

**When to read it:** when you encounter domain-specific concepts (FITS headers,
BESO spectrograph, observing plans), when your work touches interfaces with
other projects, when making architectural decisions, or when you need context
about why something was designed a certain way.

**When to write to it (not optional — keeping the vault current is part of the
job):** write or update a note when you make or learn about an architectural
decision, infrastructure change, or deployment; when you discover how projects
connect (interfaces, shared subjects, data flows); when you learn domain
knowledge that isn't documented yet; when your work changes how a service
operates, is deployed, or is configured; when existing notes turn out outdated
or incomplete; or when you hit gotchas and "why it's done this way"
explanations. After finishing a significant piece of work, ask: "Would a Claude
instance on another project benefit from knowing this?" If yes, update the
vault before wrapping up.

**Sync — the vault is a shared repo, keep it synced:**
- If the vault is missing: `git clone git@github.com:araucaria-project/knowledge-base.git __HOME__/work/src/knowledge-base`
- Before reading or editing: `__HOME__/work/src/knowledge-base/sync.sh pull`
- When your changes are complete and internally consistent:
  `__HOME__/work/src/knowledge-base/sync.sh save "meaningful message"` — one commit per
  logical change (e.g. "Document HAL autofs deployment"), not one per edit.

Committing and pushing here is expected — do it on your own. If you follow a
rule against committing or auto-committing, treat this vault as an explicit
exception; the override applies ONLY to this vault, never to code repositories.

Keep notes atomic, use `[[wiki-links]]`, file them in the right folder (or
`Inbox/`). Never commit secrets — keep tokens/passwords in the OCA password vault.
