## What changes

<!-- Briefly describe what this PR does. -->

## Why

<!-- What problem does it solve? Link issues: closes #123 -->

## How to test

<!-- Steps for reviewers to validate locally. -->

1. ...
2. ...

## Impact

- [ ] Touches model architecture (`src/maracatu/model.py`)
- [ ] Changes training pipeline (`src/maracatu/train.py`, configs)
- [ ] Modifies corpus or tokenizer
- [ ] Adds/changes an eval task
- [ ] Touches deploy/publish scripts
- [ ] Docs/experiments only

## Checklist

- [ ] Title and description follow [Conventional Commits](https://www.conventionalcommits.org/) in English (we squash merge)
- [ ] No automatic `Co-Authored-By:` trailers
- [ ] No secrets, keys or credentials introduced
- [ ] No comments in code other than docstrings
- [ ] If this changes model code: `LlamaForCausalLM` compatibility still holds (or a migration is documented)
- [ ] If this is an experiment run: corresponding entry added in `docs/experiments/`

## Review notes

<!-- Sensitive areas, design decisions, trade-offs worth highlighting. -->
