## What

<!-- A short summary of the change. -->

## Why

<!-- The problem it solves, and the concrete use case. -->

## How it was verified

<!-- The exact commands you ran, and what they showed. -->

```bash
python tests/test_router.py
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
```

## Checklist

- [ ] Focused on one change (split unrelated work into another PR)
- [ ] Rebased on the latest `main`
- [ ] Tests pass locally
- [ ] Docs or examples updated when the public API changed

Fixes #
