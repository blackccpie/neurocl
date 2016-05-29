# neurocl
Neural networks implementations

## Prerequisite:

main library dependencies:

- [Boost C++](https://github.com/boostorg)
- [VexCL](https://github.com/ddemidov/vexcl)
- [CImg](https://github.com/dtschump/CImg)

sample applications dependencies:

- [ccv](http://libccv.org)

## Building:

building ccv (used for face detection):
```
$ git clone https://github.com/liuliu/ccv.git
$ cd ccv/lib
$ ./configure
$ make
```

building neurocl (mainly header-only dependencies):
```
$ sudo apt-get install libboost-all-dev
$ git clone https://github.com/dtschump/CImg.git
$ git clone https://github.com/ddemidov/vexcl.git
$ git clone https://github.com/blackccpie/neurocl.git
$ cd neurocl
$ sh build_gcc.sh
```

## User Guide:

neurocl requires three main input files:

1. the topology description file: this is a structured text file describing the neural net layers.
```
layer:0:28x28
layer:1:6x6
layer:2:10x1
```
2. the neural net weights file: this is a binary file containing the layers weight and bias values. This file is managed internally by neurocl, but user has to specify the name of the weights file to load for training/classifying.

3. the training set description file: this is a structured text file containing a list of image sample locations along with their expected output vectors. This file is only useful for net training steps.
```
/home/my_user/train-data/sample1.bmp 0 0 0 0 0 1 0 0 0 0
/home/my_user/train-data/sample2.bmp 1 0 0 0 0 0 0 0 0 0
/home/my_user/train-data/sample3.bmp 0 0 0 0 1 0 0 0 0 0
/home/my_user/train-data/sample4.bmp 0 1 0 0 0 0 0 0 0 0
/home/my_user/train-data/sample5.bmp 0 0 0 0 0 0 0 0 0 1
...
```

neurocl main entry point is class **network_manager**:
- the network implementation backend is chosen at construction
```
neurocl::network_manager net_manager(neurocl::network_manager::NEURAL_IMPL_BNU );
```
- a given network can be loaded, given its topology and weights file names
```
net_manager.load_network( "topology.txt", "weights.bin" );
```
- once a network is loaded, it can be trained, or used for direct output computation
```
neurocl::sample sample(...);
net_manager.compute_output( sample );
```

## Architecture:

Class diagram:

![PlantUML class diagram](http://plantuml.com/plantuml/png/XP91Ri8m44NtFiM8Rbf4KBfYWxBa0XmWcfXnOc4co3QXKilTGIMnQYSDAqlVppFpvriQT0wO_BMrnrAsh7GDsosyxrTlkzrca-SVu3JNXdpBK1J2jpNvXYny2ysUh499uNrGX8pgLdmfAtIftD6NDE8s0LjI4wf2vnFvX8mrsKHLsb3P81_CwChXwMo6GVHZNFIwez9Rr1pW9-H2TP4QWVNwfvYm7Jbx1VL6OooiAZN-0kj7XSNd0fPPzdl-ttgEZfOtCfvbHV9T4jCpmD3rBzBdCKeYWeOSvgbdAH19UXFC7G00)

## Targeted platforms:

- Mac OSX
- Raspberry Pi Raspbian

## References:
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com)
- [Stanford class about sparse auto-encoders](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)

## ToRead:
- http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
-> §7.3.3 concerning matrix form

## License:

neurocl source is distributed under [MIT License](https://en.wikipedia.org/wiki/MIT_License)
