---
id: fd341cc3-b852-4d38-a4b1-7af2e8f327c1
slug: prompt-detect-cli
entity: holdingco
---

# prompt-detect

A heuristic CLI that scans a directory for files containing LLM prompt content
(system prompts, instructions, few-shot templates, prompt-injection payloads).

## Install

```sh
ln -s "$PWD/prompt_detect.py" /usr/local/bin/prompt-detect
chmod +x prompt_detect.py
```

## Usage

```sh
prompt-detect                  # scan CWD
prompt-detect path/to/dir      # scan a directory
prompt-detect file.md          # scan a single file
prompt-detect --json           # machine-readable output
prompt-detect --threshold 3    # only show possible/likely prompts
prompt-detect --top 10         # show top 10 by score
prompt-detect --all-files      # don't restrict to known text extensions
```

Exit code is `0` if any findings meet the threshold, `1` otherwise.

## How it scores

Six signal groups, each with diminishing returns past the first hit:

| signal       | weight | examples                                                  |
|--------------|--------|-----------------------------------------------------------|
| role         | 3.0    | "You are a helpful assistant", "Act as a senior engineer" |
| injection    | 2.5    | "Ignore previous instructions", "jailbreak", "DAN mode"   |
| instruction  | 2.0    | "Never reveal X", "Follow these rules", `<|system|>`      |
| fewshot      | 1.5    | `Q:` / `A:` pairs, `Input:` / `Output:` blocks            |
| api          | 1.0    | imports of anthropic/openai, `role="system"`              |
| template     | 0.8    | `{{var}}`, `{var}`, `${var}`                              |

Labels: `prompt-injection-suspect` (any injection hit) → `likely-prompt` (≥6)
→ `possible-prompt` (≥3) → `weak-signal` (≥1).

## Limits

- Reads at most 512 KiB per file.
- Skips binary files and common build/cache dirs (`.git`, `node_modules`, etc.).
- Heuristic only — review findings before acting on them.
