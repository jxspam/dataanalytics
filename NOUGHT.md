<!-- nought: generated on publish, and overwritten by the next one. -->

# dataanalytics

## Install

```bash
npm install -g @anthropic-ai/claude-code@2.1.220
```

Versions are pinned to the image nought runs this config in. On anything else you are not running the config you authored.

## Keys

- `ANTHROPIC_API_KEY` for the `anthropic/` models in this config
- `OPENAI_API_KEY` for the `openai/` models in this config

Pi and Claude Code route on the model id prefix and never fall back to another provider, so a key for a different one will not stand in.

## Run

```bash
claude --agent agent-11 -p "<your task>"
```

Run from the root of this folder: every path above is relative to it.

## Permission

This config declares a `permission:` policy, so it needs the permission extension, and the extension needs its policy file. Installed without one it defaults to asking, and `pi -p` has nobody to ask, so every tool call is refused.

```bash
mkdir -p ~/.pi/agent/extensions/pi-permission-system
cp .pi/extensions/pi-permission-system/config.json ~/.pi/agent/extensions/pi-permission-system/config.json
```

The same file sits at `.pi/extensions/pi-permission-system/config.json` for a project-scoped policy, but a project one loads only after you trust the directory, so a fresh clone reads the global one.

## Layout

```
agents-src/*.yaml   the source, edit here or in nought
chains-src/*.yaml
skills-src/*.yaml
.claude/agents/     compiled, and what Claude Code reads
manifest.json       what this config needs, for a tool to read
```

The compiled half is regenerated from the source on every publish, so do not hand edit it. It is not the authoring file with a wrapper round it either: each harness has its own dialect, and for Pi in particular lists become comma strings, `turnBudget` becomes inline JSON, and a chain becomes markdown `## <agent>` sections.
