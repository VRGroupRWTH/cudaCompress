# cudaCompress (feat. Particle Tracer)

DWT-based compression of scientific data using CUDA.

This version is based on the [cudaCompress fork of jebbstewart](https://github.com/jebbstewart/cudaCompress) which included Linux and CMake support.
The [original source code](https://github.com/m0bl0/cudaCompress) belongs to [m0bl0](https://github.com/m0bl0).

Here, slight modifications towards a more modern CMake usage and cross-compatability to Windows were made.
Additionally, a particle tracing usage case was added, featuring:
- Synthetic 4D vectorfield
- Workflow to compress 4D vector fields
- Simple Runge-Kutta 4th order integrator 

To build, clone the project then perform the following:

```bash
cd cudaCompress
mkdir build
cd build
# Use CMAKE_CUDA_ARCHITECTURES variable suited for your hardware, 86 is suited for an RTX 3090
cmake -DCMAKE_CUDA_ARCHITECTURES=86 .. 
make
```

To run the example, from the base directory (ie: cudaCompress) do the following:

```bash
./build/examples
```

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
