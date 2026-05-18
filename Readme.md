# AnyMagma

A community port of clMAGMA, supporting OpenCL 1.2+ devices and tuned clBLAST.

## Overview

This project is a future‑port of [clMAGMA](https://icl.utk.edu/magma/) maintained by **Prof. Jinchuan Tang** ([jctang@gzu.edu.cn](mailto:jctang@gzu.edu.cn)).  
It works with many OpenCL 1.2+ devices and uses a tuned version of **clBLAST** (the maintainer and his students at GZU actively contribute to clBLAST as well).

> **Note:** Some code comments are written in Chinese. If needed, use a large language model to translate them into your language.  
> Due to limited personnel and funding, LLMs may also be used to help improve the code.

The maintainer is also active in:
- **AnySparse** – a sparse library ported from clSPARSE, and  
- **AnyArray** (formerly Octave ocl extra) – providing `gpuArray`-like functionality for GNU Octave.

## Supported Algorithms

- LU, QR, and Cholesky factorizations (multi‑GPU support included)
- Hessenberg, bidiagonal, and tridiagonal reductions
- Linear solvers based on LU, QR, and Cholesky
- Eigenvalue and singular value problem solvers
- Orthogonal transformation routines

## Precisions

All four standard precisions are supported:

| Prefix | Precision          |
|--------|--------------------|
| `s`    | single (float)     |
| `d`    | double (double)    |
| `c`    | single‑complex     |
| `z`    | double‑complex     |

## Recent Updates (2026‑05‑19)

- Updated all Makefiles to support the latest clBLAST and OpenBLAS
- Fixed Python 2 legacy code → Python 3 syntax
- Many small fixes for variable initialisation
- Added support for **Windows 11**, **Ubuntu 26.04**, and **macOS 26.5**

## Building AnyMagma

> **Important:** This is a port of clMAGMA. Please read the steps below carefully – do not expect a flawless build without following them.

### 1. Choose your `make.inc`

In the root of the project folder you will find three template files:

- `make.inc.AnyMagma Apple Silicon Accelerate`
- `make.inc.AnyMagma Linux OpenBLAS`
- `make.inc.AnyMagma Windows Git Bash MingW64 OpenBLAS`

Copy or rename the one that matches your operating system to `make.inc`.

### 2. Install dependencies

#### Linux (Ubuntu 26.04 / similar)

Run the following commands:

    sudo apt update
    sudo apt install build-essential gfortran
    sudo apt install libopenblas-dev
    sudo apt install libclblast-dev
    sudo apt install opencl-headers ocl-icd-opencl-dev

#### macOS (with Apple Silicon or Intel)

First, install Xcode Command Line Tools and Homebrew (if not already present):

    xcode-select --install
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then install the required packages:

    brew install gcc openblas clblast

> **Note:** The Makefile uses Apple Clang (`cc`/`c++`) for C/C++ – only `gfortran` is needed from the `gcc` formula.

#### Windows (Git Bash + MingW64)

**Prerequisites:** Git Bash and a MingW64 toolchain are required.

1. Download the MingW64 toolchain from [winlibs](https://github.com/brechtsanders/winlibs_mingw/releases/download/16.1.0posix-14.0.0-ucrt-r1/winlibs-x86_64-posix-seh-gcc-16.1.0-mingw-w64ucrt-14.0.0-r1.7z) and unzip it.

2. Download Git Bash from [here](https://github.com/git-for-windows/git/releases/download/v2.54.0.windows.1/Git-2.54.0-64-bit.exe) and install it.

3. Open Git Bash and add the MingW64 `bin` folder to your `PATH` (adjust the path to your actual location):

       export PATH="/c/Users/HUAWEI/Documents/winlibs-x86_64-posix-seh-gcc-13.1.0-llvm-16.0.5-mingw-w64ucrt-11.0.0-r5/mingw64/bin:$PATH"

4. Set the `CLMAGMA_PATH` environment variable (points to the kernel sources):

       export CLMAGMA_PATH="/c/Users/HUAWEI/Downloads/clMagma_next-dev/clMagma_next-dev/clmagmablas/"

5. Install OpenCL ICD and headers  
   Download and install the AMD OCL‑SDK Light:  
   [OCL_SDK_Light_AMD.exe](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/download/1.0/OCL_SDK_Light_AMD.exe)  
   (Alternatively, you can use the [Khronos OpenCL‑SDK](https://github.com/KhronosGroup/OpenCL-SDK).)

6. Download clBLAST for Windows (64‑bit release) from  
   [CLBlast-1.7.0-windows-x64.7z](https://github.com/CNugteren/CLBlast/releases/download/1.7.0/CLBlast-1.7.0-windows-x64.7z)  
   and unzip it.

7. Download OpenBLAS for Windows (64‑bit) from  
   [OpenBLAS-0.3.33-x64.zip](https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33-x64.zip)  
   and unzip it.

8. Edit your `make.inc` (the one you renamed for Windows).  
   Modify the following variables to point to the actual locations on your system:

       OPENCL_DIR   = "C:/Program Files (x86)/OCL_SDK_Light"
       CLBLAST_DIR  = C:/Users/HUAWEI/Downloads/CLBlast-1.7.0-windows-x64
       OPENBLAS_DIR = C:/Users/HUAWEI/Downloads/OpenBLAS-0.3.33-x64

   (Adjust the paths according to where you unzipped the files.)

### 3. Build the library

After setting up the correct `make.inc` and installing all dependencies, run:

- **On Linux / macOS:**  

      make lib

  The compiled library will be placed in the `lib/` directory.

- **On Windows (Git Bash with MingW64):**  

      ming32-make lib

#### Kernel compilation: compile‑time vs runtime

By default, clMAGMA compiles its OpenCL kernels **at compile time** and saves them as `lib/clmagma_kernels.co`.

- **NVIDIA GPUs** can successfully generate this kernel file (tested).  
- **AMD** devices have not been fully tested.  
- **Intel** devices have very strict requirements and may fail to produce the kernel file.

If the pre‑compiled kernel file (`clmagma_kernels.co`) is available, the library searches for it at runtime via `CLMAGMA_PATH` or `LD_LIBRARY_PATH`.

**Alternative (recommended for portability):**  
If you do not want to compile kernels at compile time – for example, when you do not know which GPU will be used at runtime – simply keep a copy of the `clmagmablas` directory and point the environment variables to it.

Example for **Windows**:

    export CLMAGMA_PATH="/c/Users/HUAWEI/Downloads/clMagma_next-dev/clMagma_next-dev/clmagmablas"
    export LD_LIBRARY_PATH="/c/Users/HUAWEI/Downloads/clMagma_next-dev/clMagma_next-dev/lib:$LD_LIBRARY_PATH"

For **Linux / macOS**, use analogous paths, e.g.:

    export CLMAGMA_PATH="/path/to/your/clmagmablas"
    export LD_LIBRARY_PATH="/path/to/your/lib:$LD_LIBRARY_PATH"

Then the kernels will be loaded from source at runtime, ensuring compatibility with any OpenCL device.

## License

Refer to the original clMAGMA license and any additional notes in the source files.
