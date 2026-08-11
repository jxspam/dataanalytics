<!-- nought: generated on publish, and overwritten by the next one. -->

# dataanalytics

## Install

```bash
npm install -g @earendil-works/pi-coding-agent@0.82.1
pi install npm:pi-subagents@0.37.2
pi install npm:pi-mcp-adapter@2.15.0
pi install npm:pi-web-access@0.15.0
pi install npm:@gotgenes/pi-permission-system@24.0.0
```

`pi install`, not `npm install -g`, for the extensions. A global npm install puts one on disk without registering it in Pi's settings, so the tools it provides never appear, and the run exits 0 having quietly done nothing.

Versions are pinned to the image nought runs this config in. On anything else you are not running the config you authored.

## Keys

- `ANTHROPIC_API_KEY` for the `anthropic/` models in this config
- `OPENAI_API_KEY` for the `openai/` models in this config

Pi and Claude Code route on the model id prefix and never fall back to another provider, so a key for a different one will not stand in.

## Run

```bash
pi -p "Use the subagent tool to delegate to agent-11: <your task>"
```

Run from the root of this folder: every path above is relative to it.

Pi has no flag that selects an agent: the only thing that reads an agent definition is pi-subagents, and it is reached through the `subagent` tool. Asking for the delegation is how you name the agent.

## Permission

This config declares a `permission:` policy, so it needs the permission extension, and the extension needs its policy file. Installed without one it defaults to asking, and `pi -p` has nobody to ask, so every tool call is refused.

```bash
mkdir -p ~/.pi/agent/extensions/pi-permission-system
cp .pi/extensions/pi-permission-system/config.json ~/.pi/agent/extensions/pi-permission-system/config.json
```

The same file sits at `.pi/extensions/pi-permission-system/config.json` for a project-scoped policy, but a project one loads only after you trust the directory, so a fresh clone reads the global one.

## MCP

Needs a server for `composio`, configured in `.pi/mcp.json`, and Pi restarted after. A global `directTools: true` does not stand in for listing the tools on the agent.

## Layout

```
agents-src/*.yaml   the source, edit here or in nought
chains-src/*.yaml
skills-src/*.yaml
.pi/                compiled, and what Pi reads
manifest.json       what this config needs, for a tool to read
```

The compiled half is regenerated from the source on every publish, so do not hand edit it. It is not the authoring file with a wrapper round it either: each harness has its own dialect, and for Pi in particular lists become comma strings, `turnBudget` becomes inline JSON, and a chain becomes markdown `## <agent>` sections.
