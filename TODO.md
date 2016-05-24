# TODO - General
- [ ] Conclude with FCNN weight decay management implementation
- [ ] Update doc with xml config, picoPi2 TTS use, new network type enums...
- [ ] Optimization works:
    - [x] end major/obvious simd optimizations in the bnu fast implementation
    - [x] Try to use boost specific containers (static_vector etc...)
    - [ ] Get a better understanding of simd memory alignment constraints applied to layer sizes
    - [ ] try a fast exp(x) implementation for sigmoid function
    - [ ] Try to use boost bounded_array as ublas matrix/vector storage
    - [ ] Work with compiler flags (fast-math, unroll-loops, simd flags etc...)
- [ ] Work on a better networks class refactoring (factorization...)
- [ ] Work on CNN implementation

# TODO - Face recognition
- [ ] Canny vs Sobel proper benchmark
- [ ] Use dual Canny/Sobel nets?

# Current MNIST benchmarks (60000 samples / 10 epochs / 10 samples per batch):

| Implementation | Training time | Testing time |
| :--- | :---: | :---: |
| `Macbook Pro - Intel Core 2 Duo` | | |
| BNU_REF | 65s | 4,3s |
| BNU_FAST (SSE4.1) | 30 (x2,16) | 1,5s |
| `Raspberry Pi - Arm cortex A7` | | |
| BNU_REF | 375s | 10s |
| BNU_FAST (Neon) | 224s (x1.67) | 7,5s |
