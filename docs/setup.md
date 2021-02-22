# Setup

FastAI.jl is not registered yet since it depends on some unregistered packages (Flux.jl v0.12.0), but you can try it out using the included `Manifest.toml`. You will need Julia 1.6 for this (!) as the Manifest is not backwards compatible. You should be able to install FastAI.jl using the REPL as follows:

```julia
] clone https://github.com/lorenzoh/FastAI.jl
] activate FastAI
] instantiate
```