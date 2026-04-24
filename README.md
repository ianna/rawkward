# <img src="docs/images/rust-reverted-logo.svg"> Яawkward

_Rust-backed engine for nested, variable-length, columnar data_

# Why Rawkwrd?

**Rawkwrd** is a specialized "foundry" for the future of Awkward Array. To evolve the library while maintaining its status as a production-grade standard, we have decoupled core kernel development into this dedicated project. 

Our mission is to forge a high-performance, Rust-based engine that remains 100% structurally compatible with the existing ecosystem.

---

### 🛠️ The Foundry Approach

#### 1. Zero-Friction Portability
We replicate current kernel structures exactly to ensure **"plug-and-play"** compatibility. 
* **Direct Integration:** Once a Rust kernel is validated here, it can be copied back to the main Awkward repository with minimal refactoring.
* **Structural Parity:** By mirroring the existing architecture, we eliminate the "integration tax" usually associated with major language migrations.

#### 2. Isolated Validation (The MVP)
Rawkwrd acts as a clean-room environment to prove the Rust implementation before it touches production users.
* **Bit-for-Bit Parity:** We expose the Rust core to Python via a lean MVP to run real-world comparisons against legacy C++ kernels.
* **Performance Benchmarking:** We measure computational overhead and memory safety in isolation, ensuring the new core is faster and more robust.

#### 3. Decoupled Logic
The guiding technical principle of this project is that **kernels do not require Python.**
* **Rust as the Engine:** Rust serves as a standalone compute layer, independent of the Python interpreter’s lifecycle.
* **Strict Boundaries:** This split enforces a clean separation between the heavy-lifting heterogeneous logic and the ergonomic Python wrapper.

#### 4. Beyond Branching: The Project Split
While standard development often happens in branches, the fundamental shift to Rust requires a clean slate.
* **No Git Pollution:** Keeps the main production repository free of experimental Rust boilerplate and prototype build scripts.
* **Dedicated CI/CD:** We leverage independent, high-speed test suites without the overhead of the massive main library.
* **Architectural Freedom:** We utilize the full power of the Rust toolchain (`cargo`) from the ground up to build a modern, safe foundation.

---

> **Rawkwrd is not a fork—it is a laboratory.** We mirror the structure to ensure compatibility, use Rust to ensure performance, and keep it separate to ensure the production library remains rock-solid.
