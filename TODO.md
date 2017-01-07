# TODO - MLP
- [ ] Conclude with weight decay management implementation
- [x] Update doc with xml config, picoPi2 TTS use, new network type enums...
- [ ] Optimization works:
    - [x] End major/obvious simd optimizations in the bnu fast implementation
    - [x] Try to use boost specific containers (static_vector etc...)
    - [ ] Get a better understanding of simd memory alignment constraints applied to layer sizes
    - [ ] Try a fast exp(x) implementation for sigmoid function
    - [ ] Try to use boost bounded_array as ublas matrix/vector storage
    - [ ] Work with compiler flags (fast-math, unroll-loops, simd flags etc...)
- [ ] Work on a better networks class refactoring (factorization...)
- [ ] Work on a better solvers class refactoring (factorization...)

# TODO - CONVNET
- [ ] Optimize inverse pooling with cached pooling map
- [ ] Merge convnet implementation to Git head
- [ ] Tensor optimizations:
    - [ ] Use expression templates to combine tensor operators
    - [ ] Ublas speed improvements (cf. Boost guidelines)
    - [ ] Ublas special products (axpy)
    - [ ] Move semantics checks
    - [ ] Valgrind/kcachegrind profiling
- [x] Parallel batch training
- [ ] Confirm bias tensors initialisation
- [ ] Cross validate with similar ConvnetJS topologies
- [ ] Use "expanded" (rot/noise/trans) MNIST dataset
- [x] Introduce fan-in size layer base method
- [ ] Python:
    - [ ] Turn pyneurocl into a '.pyd'
    - [ ] Finish camera wrapper using numpy
    - [ ] Finalize autonomous pizero OCR app
- [ ] Update UML with Convnet
- [ ] Improve tensor random inits methods' names and prototypes (use fan-in denomination)
- [ ] ConvNet layers code factorization
- [ ] If needed implement Adam solver

# TODO - Face recognition
- [ ] Canny vs Sobel proper benchmark
- [ ] Use dual Canny/Sobel nets?
- [ ] Convert facecam to new training architecture

# TODO - OCR
- [ ] Use gravity center based image centering
- [ ] Work on a better autocontrast algorithm

# Current MLP-MNIST benchmarks (60000 samples / 10 epochs / 10 samples per batch):

| Implementation | Training time | Testing time |
| :--- | :---: | :---: |
| `Macbook Pro - Intel Core 2 Duo` | | |
| BNU_REF | 65s | 4,3s |
| BNU_FAST (SSE4.1) | 30 (x2,16) | 1,5s |
| `Raspberry Pi - Arm cortex A7` | | |
| BNU_REF | 375s | 10s |
| BNU_FAST (Neon) | 224s (x1.67) | 7,5s |
