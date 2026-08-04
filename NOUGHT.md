<!-- nought: generated on publish, and overwritten by the next one. -->

# dataanalytics

## Install

```bash
npm install -g @earendil-works/pi-coding-agent@0.82.1
pi install npm:pi-subagents@0.37.2
pi install npm:pi-mcp-adapter@2.15.0
pi install npm:pi-web-access@0.15.0
```

`pi install`, not `npm install -g`. A global npm install puts an extension on disk without registering it in Pi's settings, so the tools it provides never appear, and the run exits 0 having quietly done nothing.

Versions are pinned to the image nought runs this config in. On anything else you are not running the config you authored.

## Keys

- `ANTHROPIC_API_KEY` for the `anthropic/` models in this config

Pi routes on the model id prefix and never falls back to another provider, so a key for a different one will not stand in.

## Run

```bash
pi -p "Use the subagent tool to delegate to agent-11: <your task>"
```

Run it from the folder holding `.pi/`, since discovery walks up to the nearest one.

Pi has no flag that selects an agent: the only thing that reads an agent definition is pi-subagents, and it is reached through the `subagent` tool. Asking for the delegation is how you name the agent.

## Layout

```
agents-src/*.yaml   the source, edit here or in nought
chains-src/*.yaml
skills-src/*.yaml
.pi/                compiled, and what Pi reads. Regenerated on publish, so do not hand edit
manifest.json       what this config needs, for a tool to read
```

The two layers are not one file with a wrapper round it: lists become comma strings, `turnBudget` becomes inline JSON, a chain becomes markdown `## <agent>` sections. Publishing from nought rewrites `.pi/`, so there is no build step to run here.
