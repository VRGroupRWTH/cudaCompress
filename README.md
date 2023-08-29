# cudaCompress feat. cudaVectorCompress

DWT-based compression of scientific data using CUDA.
This program is tailored towards application on 3D and 4D vectorfields composed of floating point values.

This version is based on the [cudaCompress fork of jebbstewart](https://github.com/jebbstewart/cudaCompress) which included Linux and CMake support.
The [original source code](https://github.com/m0bl0/cudaCompress) belongs to [m0bl0](https://github.com/m0bl0).
This version included support to compress/decompress 3D/4D floating point vector fields using cudaCompress and incorporated some fixes for Windows compatability.

### Requirements

- C++17
- [CMake 3.15+](https://cmake.org/)
- [Git](https://git-scm.com/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Usage

```bash
cudaVectorCompress DATASET_PATH SAVE_DECOMPRESSED SAVE_INTERLEAVED [decomposition_levels (2)] [quantization_step_size (0.00136)] [compression_iterations (10)] [huffman_bits (14)]

# Compress & save the encoded dataset, then uncompress & save decoded dataset (interleaved)
cudaVectorCompress "MyDataset.raw" 1 1

# Compress & save the encoded dataset, then uncompress & save decoded dataset (not interleaved)
cudaVectorCompress "MyDataset.raw" 1 0

# Compress & save the encoded dataset, then uncompress & save decoded dataset (not interleaved)
# Also use custom compression values (here, only the huffman bits have been increased)
cudaVectorCompress "MyDataset.raw" 1 0 2 0.00136 10 16
```

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `DATASET_PATH` | - | Absolute path to the source dataset |
| `SAVE_DECOMPRESSED` | - | Whether to subsequently decode and save decoded representation |
| `decomposition_levels` | 2 | Count of decompositions before compression |
| `quantization_step_size` | 0.00136 | Size of the quantization steps used for compression |
| `compression_iterations` | 10 | Count of compression iterations |
| `huffman_bits` | 14 | Bits employed to store compressed representation |

### FAQ

- In `Debug` mode, an assertion happens upon compression.  
    This may happen due to non-power-of-2 dataset extents. This assertion won't happen in `Release` and can be circumvented by using power-of-2 datasets.
- Upon compression errors/warnings are thrown that the Huffman table design failed.  
    This may happen because of the quantization step size or the huffman bits. First, try increasing the huffman bits as it is the simplest parameter.
- How were the default values chosen?  
    The default values were taken from the sample implementations of the original author (Treib) and have not been changed.

## Installation

### Dependencies

Dependencies are automatically fetched via CMake and [CPM](https://github.com/cpm-cmake/CPM.cmake).

### Building

To build, clone the project then perform the following:

```bash
cd cudaCompress
mkdir build
cd build
# Use CMAKE_CUDA_ARCHITECTURES variable suited for your hardware, 86 is suited for an RTX 3090
cmake -DCMAKE_CUDA_ARCHITECTURES=86 .. 
make
```

You can find the correct value for `DCMAKE_CUDA_ARCHITECTURES` at, e.g., [https://arnon.dk](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
To run the example, from the base directory (ie: cudaVectorCompress) do the following:

```bash
./build/Release/cudaVectorCompress
```

## Troubleshooting

- I get the following assertion: `Assertion failed: numRows <= 1 || rowPitch <= plan->m_rowPitches[0]`  
    Not all dataset dimensions are equally safely supported by cudaCompress. This assertion typically only happens in `Debug` mode and can be ignored by using `RelWithDebInfo` / `Release` or circumvented by power-of-2-dimensions.
- I get the following warning: `WARNING: distinctSymbolCount == 9225 > 1024, huffman table design failed.`  
    In this case you need to increase the huffman bits.

## File Structure

### .cudaComp files

Holds the cudaCompressed file contents.
```bash
# myCompressedFile.cudaComp
uint:  x Dim
uint:  y Dim
uint:  z Dim
uint:  t Dim
int:   number of channels
int:   number of decomposition levels
float: quantization step size
int:   compression iterations
int:   huffman bits

for t in t Dim:
    for c in number of Channels:
        size_t: number of compressed Bytes for this channel in this timeslice
        *** compressed channel c for current timeslice t ***

EOF
```

### .raw files

Hold uncompressed data. If no supplementary `<filename>_dims.raw` file exists, `*.raw` files are expected to look like this:
```bash
# myFile.raw
uint:  x Dim
uint:  y Dim
uint:  z Dim
uint:  t Dim
*** Dataset ***
EOF
```

Else, they are expected to look like this:
```bash
# myFile_dims.raw
uint:  x Dim
uint:  y Dim
uint:  z Dim
uint:  t Dim
EOF

# myFile.raw
*** Dataset ***
EOF
```

Dataset is expected to be either interleaved or not.
If it is interleaved, the components are interpreted as "xyz, xyz, xyz, ...".
Else, they are read as 3 concatenated datasets as "xxx..., yyy..., zzz...".  

# Original Readme

This is the source code for the cudaCompress library, which was used in the papers

Interactive Editing of GigaSample Terrain Fields
by Marc Treib, Florian Reichl, Stefan Auer, and R�diger Westermann
published in Computer Graphics Forum 31,2 (Proc. Eurographics 2012)
http://wwwcg.in.tum.de/research/research/publications/2012/interactive-editing-of-gigasample-terrain-fields.html

Turbulence Visualization at the Terascale on Desktop PCs
by Marc Treib, Kai B�rger, Florian Reichl, Charles Meneveau, Alex Szalay, and R�diger Westermann
published in IEEE Transactions on Computer Graphics and Visualization (Proc. IEEE Scientific Visualization 2012)
http://wwwcg.in.tum.de/research/research/publications/2012/turbulence-visualization-at-the-terascale-on-desktop-pcs.html

This code is provided under the terms of the MIT license (see the file LICENSE) with the exception of the subfolders src/cudaCompress/reduce and src/cudaCompress/scan, which contain code adapted from CUDPP (http://cudpp.github.io) and come with a separate license (see file license.txt in the subfolders).

It was developed and tested on Windows 7/8 x64 with Visual Studio 2012 using CUDA 5.5.
To run it, you need at least a Fermi-class GPU (NVIDIA GeForce GTX 4xx or later).
If you have any questions or problems to get it running, contact me at treib@tum.de.

If you use this code in a scientific publication, please cite the appropriate paper(s) above.
If you use it for anything else, it would be nice if you could drop me a line at treib@tum.de and tell me about it!
