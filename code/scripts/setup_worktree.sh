#!/usr/bin/env bash
set -euo pipefail

# Links a fresh worktree to the main checkout's data and to a per-worktree
# output folder. Works for both tools:
# - Codex: the worktree already exists; CODEX_WORKTREE_PATH and
#   CODEX_SOURCE_TREE_PATH are set to absolute paths.
# - Claude Code: runs as a WorktreeCreate hook, which replaces Claude's default
#   `git worktree add`. The script creates the worktree, reads its name from the
#   JSON on stdin, and must print the worktree path as the last stdout line.
#   Everything else goes to stderr.

if [[ -n "${CODEX_WORKTREE_PATH:-}" ]]; then
    source_tree="$CODEX_SOURCE_TREE_PATH"
    worktree="$CODEX_WORKTREE_PATH"
else
    source_tree="$CLAUDE_PROJECT_DIR"
    name=$(jq -r .name)
    worktree="$source_tree/.claude/worktrees/$name"

    # Same as Claude's default: reopen an existing worktree with this name.
    if [[ -d "$worktree" ]]; then
        echo "$worktree"
        exit 0
    fi

    # Same as Claude's default: new branch worktree-<name> from origin's default branch.
    git -C "$source_tree" fetch --quiet origin >&2 || true
    base=$(git -C "$source_tree" rev-parse --verify --quiet origin/HEAD || git -C "$source_tree" rev-parse HEAD)
    git -C "$source_tree" worktree add -b "worktree-$name" "$worktree" "$base" >&2
fi

# Git assigns a unique ID even when worktrees have identical folder names.
worktree_id=$(basename "$(git -C "$worktree" rev-parse --absolute-git-dir)")
output_dir="$source_tree/output/$worktree_id"

mkdir -p "$output_dir"
ln -s "$source_tree/data" "$worktree/data"
ln -s "$output_dir" "$worktree/output"

if [[ -z "${CODEX_WORKTREE_PATH:-}" ]]; then
    echo "$worktree"
fi
