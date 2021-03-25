# Setup

FastAI.jl is not registered yet since it depends on some unregistered packages (Flux.jl v0.12.0 and a Metalhead.jl PR), but you can try it out by installing these manually. You should be able to install FastAI.jl using the REPL as follows:

```julia
] clone https://github.com/lorenzoh/FastAI.jl
] activate FastAI
] add Flux#master https://github.com/darsnack/Metalhead.jl#darsnack/vision-refactor
```

FastAI.jl also defines [Makie.jl] plotting recipes to visualize data. If you want to use them, you'll have to install and one of the Makie.jl backends [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl), [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl) or [WGLMakie.jl](https://github.com/JuliaPlots/WGLMakie.jl). For example:

```julia
] add CairoMakie
```