# Setup

FastAI.jl is a **Julia** package. You can download Julia from the [official website](http://localhost:8000/docs/setup.md.html). You can install FastAI.jl like any other Julia package using the REPL as follows.

```julia
using Pkg
Pkg.add(Pkg.PackageSpec(url="https://github.com/FluxML/FastAI.jl"))
```

**Plotting** FastAI.jl also defines [Makie.jl](https://github.com/JuliaPlots/Makie.jl) plotting recipes to visualize data. If you want to use them, you'll have to install and one of the Makie.jl backends [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl), [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl) or [WGLMakie.jl](https://github.com/JuliaPlots/WGLMakie.jl). For example:

```julia
using Pkg
Pkg.add("CairoMakie")
```

**Colab** If you don't have access to a GPU or want to try out FastAI.jl without installing Julia, try out [this FastAI.jl Colab notebook](https://colab.research.google.com/gist/lorenzoh/2fdc91f9e42a15e633861c640c68e5e8). We're working on adding a "Launch Colab" button to every documentation page based off a notebook file, but for now you can copy the code over manually.

**Pretrained models** To use pretrained vision models, you currently have to install a WIP branch of Metalhead.jl:

```julia
using Pkg
Pkg.add(Pkg.PackageSpec(url="https://github.com/darsnack/Metalhead.jl", rev="darsnack/vision-refactor")
```

**Threaded data loading** To make use of multi-threaded data loading, you need to start Julia with multiple threads, either with the `-t auto` commandline flag or by setting the environment variable `JULIA_NUM_THREADS`. See the [IJulia.jl documentation](https://julialang.github.io/IJulia.jl/dev/manual/installation/#Installing-additional-Julia-kernels) for instructions on setting these for Jupyter notebook kernels.