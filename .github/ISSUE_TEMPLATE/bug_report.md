---
name: Bug report
about: A comparison produced a wrong, missing, or misclassified result
title: "[bug] "
labels: bug
---

## Summary

A clear description of what went wrong.

## Reproduction

The exact command line you ran:

```bash
onnx-graph-diff LEFT.onnx RIGHT.onnx --method fingerprint
```

Please attach the two ONNX models, or a minimal script that generates them.

## Printed JSON summary

```json
paste the summary printed to stdout here
```

## Expected vs. actual

- **Expected:** what conclusion / regions you expected.
- **Actual:** what the tool reported (proven / structural / rejected / unresolved).

## Environment

- `onnx-graph-diff` version:
- Python version:
- `onnx` version:
- OS:
