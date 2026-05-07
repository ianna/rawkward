# Contributing to ЯAwkward

Thank you for your interest in contributing to ЯAwkward!

ЯAwkward is a Rust‑based re‑implementation of the Awkward Array data model.
We welcome contributions in Rust, Python, GPU kernels, Arrow integration,
documentation, testing, and design discussions.

---

## Getting Started

1. Install Rust (stable) and maturin.
2. Clone the repository:
   git clone https://github.com/ianna/rawkward
3. Build the Rust library:
   cargo build
4. Build the Python bindings:
   maturin develop --features python

---

## Contribution Areas

### Rust Core
- Implementing new Content layouts
- Improving invariants and type safety
- Kernel development (slice, broadcast, reduce)

### GPU Backend
- CUDA or wgpu kernels
- Memory transfer logic
- Kernel dispatch

### Python Bindings
- PyO3 interface
- API parity with Awkward v2
- Zero‑copy conversions

### Arrow Integration
- Buffer wrappers
- Schema conversions
- Zero‑copy interoperability

### Testing
- Parity tests (Rust vs Python Awkward)
- Fuzzing
- Performance regression tests

---

## Code Style

- Use `rustfmt` for Rust code.
- Use `ruff` for Python code.
- Prefer explicit invariants over implicit assumptions.
- Avoid unsafe Rust unless absolutely necessary.

---

## Pull Requests

1. Fork the repo.
2. Create a feature branch.
3. Add tests for new functionality.
4. Submit a PR with a clear description.

---

## Reporting Issues

Please include:
- A minimal reproducible example
- Expected vs actual behavior
- Version information

---

Thank you for helping build the future of Awkward Array!
