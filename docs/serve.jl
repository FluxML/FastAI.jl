using Pollen

# using MyPackage


# Create Project
project = include("project.jl")


@info "Rewriting documents..."
Pollen.rewritesources!(project)

Pollen.serve(project, lazy=true, format = Pollen.JSON())
