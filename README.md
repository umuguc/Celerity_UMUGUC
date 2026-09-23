<p align="center">
<img src="docs/celerity_logo.png" alt="Celerity Logo">
</p>

# Celerity UMUGUC

This repository is a fork of the [Celerity runtime](https://github.com/celerity/celerity-runtime), based on Celerity 0.7.1 and developed in the UMUGUC project.
It adds **transparent data compression for buffers**: a buffer stores its data in a compressed type, and kernels and host tasks work on the uncompressed values.
Everything else behaves like Celerity `master`, so the upstream documentation below still applies.

## Differences from Celerity `master`

- **Compressed buffers.** `celerity::buffer` takes the compression as an optional third template parameter, `buffer<T, Dims, Compression>`.
  Without it, a buffer behaves exactly as before (`compression::uncompressed`).
- **Three compression categories** (`include/compression.h`), which decide where the data is uncompressed:
  - `element_wise`: every accessor read decompresses and every write compresses a single element. The accessor is used as usual.
  - `local_memory`: `acc.decompress_data(item, allocator, chunk)` decompresses a chunk into work-group local memory, and recompresses it at the end of its scope for writing accessors.
  - `global_memory`: `acc.decompress_data(item, chunk)` decompresses a chunk into a per-device scratch copy in global memory that all work groups share. A per-chunk state tracker makes sure a chunk is decompressed and compressed only once at a time. Such a buffer needs `buffer.init(queue)` before its first use.
- **Compression algorithms** (`include/compression_algorithms.h`): `quantization_compression`, `conversion_compression`, `point_cloud_compression`, `point_cloud_with_dep_compression` and `z_curve_hybrid_compression`.
  New algorithms are added by specializing `compressed<>` and wrapping it in a `compression_object_skeleton`.
- **Host tasks** read compressed buffers with `acc.decompress_data()`, which returns all values uncompressed.
- **Runtime changes needed by the compression:**
  - `chunk` carries the work-group size (`local_range`), so range mappers can map whole work groups.
  - New helper `detail::get_nd_index`.
  - The runtime exposes `get_local_nid()`, `get_num_nodes()` and `get_num_local_devices()`, which compressed buffers use to allocate one scratch and state tracking row per device.
- **Examples:** `wave_sim_compression` (element-wise, local and global memory variants and a specialized compression) and `nd_range_wave_sim`.
- **Tests:** `test/compression_algorithm_tests.cc` and `test/compression_interface_tests.cc`.

```cpp
// Points stored as half-precision offsets, decompressed into global memory on demand
using point_compression = celerity::point_cloud_compression<Point, sycl::vec<sycl::half, 3>, celerity::compression_category::global_memory>;

point_compression compression{Point{x_min, y_min, 0}, block_size};
celerity::buffer<Point, 3, point_compression> points({width, height, points_per_block}, compression);
points.init(queue);

queue.submit([&](celerity::handler& cgh) {
	celerity::accessor acc{points, cgh, celerity::access::one_to_one{}, celerity::read_only};
	cgh.parallel_for(celerity::nd_range<3>{points.get_range(), {1, 1, points_per_block}}, [=](celerity::nd_item<3> item) {
		auto block = acc.decompress_data(item); // uncompressed view of this work group's chunk
		// ... block[{x, y, i}] ...
	});
});
```
---

# Celerity Runtime — [![CI Workflow](https://github.com/celerity/celerity-runtime/actions/workflows/celerity_ci.yml/badge.svg)](https://github.com/celerity/celerity-runtime/actions/workflows/celerity_ci.yml) [![Coverage Status](https://coveralls.io/repos/github/celerity/celerity-runtime/badge.svg?branch=master)](https://coveralls.io/github/celerity/celerity-runtime?branch=master) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/celerity/celerity-runtime/blob/master/LICENSE) [![Semver 2.0](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/spec/v2.0.0.html) [![PRs # Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/celerity/celerity-runtime/blob/master/CONTRIBUTING.md)

The Celerity distributed runtime and API aims to bring the power and ease of
use of [SYCL](https://sycl.tech) to multi-GPU systems and distributed memory 
clusters, with transparent scaling.

> If you want a step-by-step introduction on how to set up dependencies and
> implement your first Celerity application, check out the
> [tutorial](docs/tutorial.md)!

## Overview

Programming modern accelerators is already challenging in and of itself.
Combine it with the distributed memory semantics of a cluster, and the
complexity can become so daunting that many leave it unattempted. Celerity
wants to relieve you of some of this burden, allowing you to target
accelerator clusters with programs that look like they are written for a
single device.

### High-level API based on SYCL

Celerity makes it a priority to stay as close to the SYCL API as possible. If
you have an existing SYCL application, you should be able to migrate it to
Celerity without much hassle. If you know SYCL already, this will probably
look very familiar to you:

```cpp
celerity::buffer<float> buf(celerity::range(1024));
queue.submit([&](celerity::handler& cgh) {
    celerity::accessor acc(buf, cgh,
        celerity::access::one_to_one(),               // 1
        celerity::write_only, celerity::no_init);
    cgh.parallel_for(
        celerity::range(1024),                        // 2
        [=](celerity::item<1> item) {                 // 3
            acc[item] = sycl::sin(item[0] / 1024.f);  // 4
        });
});
```

1. Provide a [range-mapper](docs/range-mappers.md) to tell Celerity which
   parts of the buffer will be accessed by the kernel.

2. Submit a kernel to be executed by 1024 parallel _work items_. This kernel
   may be split across any number of nodes.

3. Kernels can be expressed as C++ lambda functions, just like in SYCL. In
   fact, no changes to your existing kernels are required.

4. Access your buffers as if they reside on a single device -- even though
   they might be scattered throughout the cluster.

### Run it like any other MPI application

The kernel shown above can be run on a single GPU, just like in SYCL, on 
multiple GPUs attached to a single shared memory system, or on a whole cluster
 -- without having to change anything about the program itself.

On a cluster with 2 nodes and a single GPU each, `mpirun -n 2 ./my_example`
might have the first GPU compute the range `0-512` of the kernel, while the 
second one computes `512-1024`. However, as the user, you don't have to care how
exactly your computation is being split up.

By default, a Celerity application will use all the attached GPUs, i.e. running
`./my_example` on a machine with 4 GPUs will automatically use all of them.

To see how you can use the result of your computation, look at some of our
fully-fledged [examples](examples), or follow the [tutorial](docs/tutorial.md)!

## Building Celerity

Celerity uses CMake as its build system. The build process itself is rather
simple, however you have to make sure that you have a few dependencies
installed first.

### Dependencies

- A supported SYCL implementation, either
    - [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp),
    - [DPC++](https://github.com/intel/llvm), or
    - [SimSYCL](https://github.com/celerity/SimSYCL)
- [CMake](https://www.cmake.org) (3.13 or newer)
- A C++20 compiler
- [*optional*] An MPI 2 implementation (tested with OpenMPI 4.0, MPICH 3.3 should work as well)

See the [platform support guide](docs/platform-support.md) on which library and OS versions are supported and
automatically tested.

Building can be as simple as calling `cmake && make`, depending on your setup
you might however also have to provide some library paths etc.
See our [installation guide](docs/installation.md) for more information.

The runtime comes with several [examples](examples) that can be used as a starting
point for developing your own Celerity application. All examples will also be built
automatically in-tree when the `CELERITY_BUILD_EXAMPLES` CMake option is set
(true by default).

## Using Celerity as a Library

Simply run `make install` (or equivalent, depending on build system) to copy
all relevant header files and libraries to the `CMAKE_INSTALL_PREFIX`. This
includes a CMake [package configuration file](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#package-configuration-file)
which is placed inside the `lib/cmake/Celerity` directory. You can then use
`find_package(Celerity CONFIG)` to include Celerity into your CMake project.
Once included, you can use the `add_celerity_to_target(TARGET target SOURCES source1 source2...)`
function to set up the required dependencies for a target (no need to link manually).

## Running a Celerity Application

When running on a single machine, simply execute your application as you
normally would -- no special invocation required. By default, the runtime
will then attempt use all available GPUs. A simple way of limiting this,
e.g. for benchmarking, is to use vendor-specific environment variables such
as `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or `ONEAPI_DEVICE_SELECTOR`.

When targeting distributed memory clusters, a Celerity application can be 
executed like any other MPI-based application (i.e., using `mpirun` or equivalent).

There are also [several environment variables](docs/configuration.md) which influence
Celerity's runtime behavior. Tweaking these variables can be useful to tailor
performance to specific systems, as well as for debugging Celerity applications.
