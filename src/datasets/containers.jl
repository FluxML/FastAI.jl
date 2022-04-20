# FileDataset

function FileDataset(dir, pattern = "*")
    return rglob(pattern, string(dir))
end

pathparent(p::String) = splitdir(p)[1]
pathname(p::String) = splitdir(p)[2]

# File utilities

"""
    rglob(filepattern, dir = pwd(), depth = 4)

Recursive glob up to 6 layers deep.
"""
function rglob(filepattern = "*", dir = pwd(), depth = 4)
    patterns = [
        "$filepattern",
        "*/$filepattern",
        "*/*/$filepattern",
        "*/*/*/$filepattern",
        "*/*/*/*/$filepattern",
        "*/*/*/*/*/$filepattern",
    ]
    return vcat([glob(pattern, dir) for pattern in patterns[1:depth]]...)
end


"""
    loadfile(file)

Load a file from disk into the appropriate format.
"""
function loadfile(file::String)
    if isimagefile(file)
        # faster image loading
        return FileIO.load(file)
    elseif endswith(file, ".csv")
        return DataFrame(CSV.File(file))
    else
        return FileIO.load(file)
    end
end

loadfile(file::AbstractPath) = loadfile(string(file))


#TableDataset

struct TableDataset{T}
    table::T #Should implement Tables.jl interface
    TableDataset{T}(table::T) where {T} =
        Tables.istable(table) ? new{T}(table) :
        error("Object doesn't implement Tables.jl interface")
end

TableDataset(table::T) where {T} = TableDataset{T}(table)
TableDataset(path::AbstractPath) = TableDataset(DataFrame(CSV.File(path)))

function LearnBase.getobs(dataset::FastAI.Datasets.TableDataset, idx)
    if Tables.rowaccess(dataset.table)
        row, _ = Iterators.peel(Iterators.drop(Tables.rows(dataset.table), idx - 1))
        return row
    elseif Tables.columnaccess(dataset.table)
        colnames = Tables.columnnames(dataset.table)
        rowvals = [Tables.getcolumn(dataset.table, i)[idx] for i = 1:length(colnames)]
        return (; zip(colnames, rowvals)...)
    else
        error(
            "The Tables.jl implementation used should have either rowaccess or columnaccess.",
        )
    end
end

function LearnBase.nobs(dataset::TableDataset)
    if Tables.columnaccess(dataset.table)
        return length(Tables.getcolumn(dataset.table, 1))
    elseif Tables.rowaccess(dataset.table)
        return length(Tables.rows(dataset.table)) # length might not be defined, but has to be for this to work.
    else
        error(
            "The Tables.jl implementation used should have either rowaccess or columnaccess.",
        )
    end
end

LearnBase.getobs(dataset::TableDataset{<:DataFrame}, idx) = dataset.table[idx, :]
LearnBase.nobs(dataset::TableDataset{<:DataFrame}) = nrow(dataset.table)

LearnBase.getobs(dataset::TableDataset{<:CSV.File}, idx) = dataset.table[idx]
LearnBase.nobs(dataset::TableDataset{<:CSV.File}) = length(dataset.table)


# ## Tests

@testset "TableDataset" begin
    @testset "TableDataset from rowaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = false
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = true

        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test all(getobs(td, 1) .== [1, 4.0, "7"])
        @test nobs(td) == 3
    end

    @testset "TableDataset from columnaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = true
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = false

        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test [data for data in getobs(td, 2)] == [2, 5.0, "8"]
        @test nobs(td) == 3

        @test getobs(td, 1) isa NamedTuple
    end

    @testset "TableDataset from DataFrames" begin
        testtable = DataFrame(
            col1 = [1, 2, 3, 4, 5],
            col2 = ["a", "b", "c", "d", "e"],
            col3 = [10, 20, 30, 40, 50],
            col4 = ["A", "B", "C", "D", "E"],
            col5 = [100.0, 200.0, 300.0, 400.0, 500.0],
            split = ["train", "train", "train", "valid", "valid"],
        )
        td = TableDataset(testtable)
        @test td isa TableDataset{<:DataFrame}

        @test [data for data in getobs(td, 1)] == [1, "a", 10, "A", 100.0, "train"]
        @test nobs(td) == 5
    end

    @testset "TableDataset from CSV" begin
        open("test.csv", "w") do io
            write(io, "col1,col2,col3,col4,col5, split\n1,a,10,A,100.,train")
        end
        testtable = CSV.File("test.csv")
        td = TableDataset(testtable)
        @test td isa TableDataset{<:CSV.File}
        @test [data for data in getobs(td, 1)] == [1, "a", 10, "A", 100.0, "train"]
        @test nobs(td) == 1
        rm("test.csv")
    end
end
