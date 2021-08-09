# Development workflow

FastAI.jl, as a end-user friendly umbrella package, puts importance on helpful documentation. To make it easier to write documentation and see the results interactively, you can follow this guide.

## Setup

You'll have to do the following:

**Fork FastAI.jl and add it as a `dev` dependency.** You can fork it from [the GitHub repository](https://github.com/FluxML/FastAI.jl). Then use `Pkg` to add your fork to your Julia environment:

```julia
using Pkg
Pkg.develop(url="https://github.com/<myusername>/FastAI.jl.git")
```

**Activate the documentation environment and install the dependencies.** You can find the folder that FastAI.jl was cloned to using `using FastAI; pkgdir(FastAI)`. In a Julia session, change the current directory to that path, activate the `docs/` environment and install unregistered dependencies:

```julia
using FastAI, Pkg

cd(pkgdir(FastAI))
Pkg.activate("./docs/")
Pkg.add(url="https://github.com/lorenzoh/Pollen.jl")
Pkg.add(url="https://github.com/lorenzoh/LiveServer.jl")
Pkg.instantiate()
```

Finally you can start the development server which will serve the documentation locally and reload any changes you make:

```julia
include("./docs/serve.jl")
```

On subsequent runs, it'll be enough to activate the environment and include the startup file:

```julia
using FastAI, Pkg
Pkg.activate(joinpath(pkgdir(FastAI), "docs"))
include(joinpath(pkgdir(FastAI), "docs", "serve.jl"))
```

### Notes

For performance reasons, the development server will only build each page once you open it in the browser, you might have to refresh the tab after a few seconds. The terminal output will show when a page is being built; for documentation pages that have a lot of code cells that need to be run, it can take some time for the page to be built. If any changes are made to the source file of the documentation package, the page will automatically be rebuilt and the tab reloads. 

## Adding documentation

### Adding source files

Documentation pages correspond to a Markdown `.md` or Jupyter Notebook `.ipynb` file that should be stored in the `docs/` folder. 

- Jupyter Notebooks should be used when they use resources that are not available on the GitHub CI, like a GPU needed for training. You should run them locally and the outputs will be captured and inserted into the HTML page.
- Markdown documents should be preferred for everything else, as they allow the code examples to be run on the GitHub CI, meaning they'll stay up-to-date unlike a notebook that has to be manually rerun.

Both formats support the [Markdown syntax of Publish.jl](https://michaelhatherly.github.io/Publish.jl/dev/docs/syntax.html) and in markdown files the [cell syntax of Publish.jl](https://michaelhatherly.github.io/Publish.jl/dev/docs/cells.html) can be used to mark code cells. These will be run and the output is inserted into the HTML page.

### Linking to documentation 

For a new documentation file to be discoverable, you have to add an entry to the nested Markdown list in `toc.md`, which corresponds to the sidebar in the documentation (Updating the sidebar currently requires interrupting and reincluding the file that starts the development server).

Documentation pages can also link to each other using standard Markdown link syntax.

### Referencing code symbols

Symbols like `fitonecycle!` can be referenced by using the cross-referencing syntax ```[`fitonecycle!`](#) ``` which will link to and create a reference page from the symbol's docstrings. It will also be added as an entry on the references page.