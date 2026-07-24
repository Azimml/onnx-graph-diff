# Security Policy

## Reporting a Vulnerability

If you discover a security issue in onnx-graph-diff (for example, a crafted ONNX
model that causes unsafe file access during parsing, or a prompt-injection path
in the LLM-assisted matching routes), please open a private security advisory or
an issue with a reproduction. Do not include real API keys in reports.

## Notes

- LLM API keys are read from environment variables only and are never logged.
- The tool never executes models; it performs static graph analysis.
