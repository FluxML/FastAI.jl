# Setup

FastAI.jl is not registered yet since it depends on some unregistered packages (Flux.jl v0.12.0), but you can try it out using the included `Manifest.toml`. You will need Julia 1.6 for this (!) as the Manifest is not backwards compatible. You should be able to install FastAI.jl using the REPL as follows:

```julia
] clone https://github.com/FluxML/FastAI.jl
] activate FastAI
] instantiate
```

FastAI.jl also defines [Makie.jl](https://github.com/JuliaPlots/Makie.jl) plotting recipes to visualize data. If you want to use them, you'll have to install and one of the Makie.jl backends [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl), [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl) or [WGLMakie.jl](https://github.com/JuliaPlots/WGLMakie.jl). For example:

```julia
] add CairoMakie
```