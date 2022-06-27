using PackageCompiler

PackageCompiler.create_sysimage(
    ["CUDA", "FastAI", "FastVision", "FastMakie", "FastTabular", "Flux", "MLUtils", "FluxTraining"],
    sysimage_path = joinpath(@__DIR__, "sys.so"),
    precompile_execution_file=joinpath(@__DIR__, "workload.jl"),
    project=@__DIR__,
)
