# Setup

FastAI.jl is a **Julia** package. You can download Julia from the [official website](http://localhost:8000/docs/setup.md.html).

**FastAI.jl** is not registered yet, but you can try it out by installing it manually. You should be able to install FastAI.jl using the REPL as follows (The package mode in the REPL can be entered by typing `]`).

```julia
pkg> add https://github.com/FluxML/FastAI.jl
```

**Plotting** FastAI.jl also defines [Makie.jl](https://github.com/JuliaPlots/Makie.jl) plotting recipes to visualize data. If you want to use them, you'll have to install and one of the Makie.jl backends [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl), [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl) or [WGLMakie.jl](https://github.com/JuliaPlots/WGLMakie.jl). For example:

```julia
pkg> add CairoMakie
```

**Pretrained models** To use pretrained vision models, you currently have to install a WIP branch of Metalhead.jl:

```julia
pkg> add https://github.com/darsnack/Metalhead.jl#darsnack/vision-refactor
```

**Threaded data loading** To make use of multi-threaded data loading, you need to start Julia with multiple threads, either with the `-t auto` commandline flag or by setting the environment variable `JULIA_NUM_THREADS`. See the [IJulia.jl documentation](https://julialang.github.io/IJulia.jl/dev/manual/installation/#Installing-additional-Julia-kernels) for instructions on setting these for Jupyter notebook kernels.