# Setup

FastAI.jl is not registered yet, but you can try it out by installing it manually. You should be able to install FastAI.jl using the REPL as follows (The package mode in the REPL can be entered by typing `]`).

```julia
pkg> add https://github.com/FluxML/FastAI.jl
```

FastAI.jl also defines [Makie.jl](https://github.com/JuliaPlots/Makie.jl) plotting recipes to visualize data. If you want to use them, you'll have to install and one of the Makie.jl backends [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl), [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl) or [WGLMakie.jl](https://github.com/JuliaPlots/WGLMakie.jl). For example:

```julia
pkg> add CairoMakie
```

To use pretrained vision models, you currently have to install a WIP branch of Metalhead.jl:

```julia
pkg> add https://github.com/darsnack/Metalhead.jl#darsnack/vision-refactor
```