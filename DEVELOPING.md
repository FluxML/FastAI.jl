# Developing

This guide contains information important when developing FastAI.jl. Concretely:

- how to set up a local development environment
- how to run the tests
- how to preview the documentation locally

## Setting up FastAI.jl locally for development

**Fork FastAI.jl and add it as a `dev` dependency.** You can fork it from [the GitHub repository](https://github.com/FluxML/FastAI.jl). Then use `Pkg` to add the fork to your Julia environment:

```julia
using Pkg
Pkg.develop(url="https://github.com/<myusername>/FastAI.jl.git")
```

You should now be able to import FastAI (`using FastAI`) in Julia. If you are using [Revise.jl](https://github.com/timholy/Revise.jl), any changes you make to its source code will also be reflected in your interactive sessions.


## Running the tests

Like any Julia package, you can run the entire test suite in an isolated environment using `Pkg.test`:

```julia
using Pkg
Pkg.test("FastAI")
``` 

When developing, however, it can be helpful to repeatedly rerun parts of the tests. FastAI.jl uses [ReTest.jl](https://github.com/JuliaTesting/ReTest.jl) to set up tests which makes it possible to run subsets of tests or only tests that have not previously succeeded.

First, activate the test environment and install its dependencies:

```julia
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "FastAI", "test"))
Pkg.instantiate()
```

Then, you can run the test suite or subsets of it:

```julia
using FastAI, ReTest

FastAI.runtests()  # full test suite
FastAI.runtests("block")  # all tests containing `"block"`
FastAI.runtests([ReTest.fail, ReTest.not(ReTest.pass)])  # run only tests that have not been run or have failed previously

```


## Local documentation preview

FastAI.jl uses [Pollen.jl](https://github.com/lorenzoh/Pollen.jl) as its documentation system, which allows you to preview documentation locally.

First, activate the documentation environment and install its dependencies:

```julia
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "FastAI", "docs"))
Pkg.add([
    Pkg.PackageSpec(url="https://github.com/c42f/JuliaSyntax.jl"),
    Pkg.PackageSpec(url="https://github.com/lorenzoh/ModuleInfo.jl"),
    Pkg.PackageSpec(url="https://github.com/lorenzoh/Pollen.jl", rev="main"),
])
```

Now you can build the documentation locally, giving you a preview at [http://localhost:3000](http://localhost:3000). Using the `lazy = true` will build pages lazily only once you request them on the website, which reduces the build time when you only care about specific pages.


```julia
using FastAI, Pollen
Pollen.servedocs(@__MODULE__, FastAI, lazy = true)
```

### Adding documentation pages files

Documentation pages correspond to a Markdown `.md` or Jupyter Notebook `.ipynb` file that should be stored in the `docs/` folder. If a document should show up in the left sidebar of the docs page, add an entry to `FastAI/docs/toc.json`.

- Jupyter Notebooks should be used when they use resources that are not available on the GitHub CI, like a GPU needed for training. You should run them locally and the outputs will be captured and inserted into the HTML page.
- Markdown documents should be preferred for everything else, as they allow the code examples to be run on the GitHub CI, meaning they'll stay up-to-date unlike a notebook that has to be manually rerun.

Both formats support the [Markdown syntax of Publish.jl](https://michaelhatherly.github.io/Publish.jl/dev/docs/syntax.html) and in markdown files the [cell syntax of Publish.jl](https://michaelhatherly.github.io/Publish.jl/dev/docs/cells.html) can be used to mark code cells. These will be run and the output is inserted into the HTML page.

### Linking to documentation 

For a new documentation file to be discoverable, you have to add an entry to the nested Markdown list in `toc.md`, which corresponds to the sidebar in the documentation (Updating the sidebar currently requires interrupting and reincluding the file that starts the development server).

Documentation pages can also link to each other using standard Markdown link syntax.

### Referencing code symbols

Symbols like `fitonecycle!` can be referenced by using the cross-referencing syntax ```[`fitonecycle!`](#) ``` which will link to and create a reference page from the symbol's docstrings. It will also be added as an entry on the references page.