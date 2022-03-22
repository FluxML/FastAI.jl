using Pkg
using Pollen

# Create target folder
DIR = abspath(mkpath(ARGS[1]))


# Create Project
project = include("project.jl")


@info "Rewriting documents..."
Pollen.rewritesources!(project)

@info "Writing to disk at \"$DIR\"..."
builder = Pollen.FileBuilder(
    Pollen.JSON(),
    DIR,
)
Pollen.build(
    builder,
    project,
)
