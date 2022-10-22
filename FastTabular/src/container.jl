
struct TableDataset{T}
    table::T #Should implement Tables.jl interface
    function TableDataset{T}(table::T) where {T}
        Tables.istable(table) ? new{T}(table) :
        error("Object doesn't implement Tables.jl interface")
    end
end

TableDataset(table::T) where {T} = TableDataset{T}(table)
TableDataset(path::AbstractPath) = TableDataset(DataFrame(CSV.File(path)))

function Base.getindex(dataset::TableDataset, idx)
    if Tables.rowaccess(dataset.table)
        row, _ = Iterators.peel(Iterators.drop(Tables.rows(dataset.table), idx - 1))
        return row
    elseif Tables.columnaccess(dataset.table)
        colnames = Tables.columnnames(dataset.table)
        rowvals = [Tables.getcolumn(dataset.table, i)[idx] for i in 1:length(colnames)]
        return (; zip(colnames, rowvals)...)
    else
        error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end

function Base.length(dataset::TableDataset)
    if Tables.columnaccess(dataset.table)
        return length(Tables.getcolumn(dataset.table, 1))
    elseif Tables.rowaccess(dataset.table)
        return length(Tables.rows(dataset.table)) # length might not be defined, but has to be for this to work.
    else
        error("The Tables.jl implementation used should have either rowaccess or columnaccess.")
    end
end

Base.getindex(dataset::TableDataset{<:DataFrame}, idx) = dataset.table[idx, :]
Base.length(dataset::TableDataset{<:DataFrame}) = nrow(dataset.table)

Base.getindex(dataset::TableDataset{<:CSV.File}, idx) = dataset.table[idx]
Base.length(dataset::TableDataset{<:CSV.File}) = length(dataset.table)

# ## Tests

@testset "TableDataset" begin
    @testset "TableDataset from rowaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = false
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = true

        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test all(td[1] .== [1, 4.0, "7"])
        @test length(td) == 3
    end

    @testset "TableDataset from columnaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = true
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = false

        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test [data for data in td[2]] == [2, 5.0, "8"]
        @test length(td) == 3

        @test td[1] isa NamedTuple
    end

    @testset "TableDataset from DataFrames" begin
        testtable = DataFrame(col1 = [1, 2, 3, 4, 5],
                              col2 = ["a", "b", "c", "d", "e"],
                              col3 = [10, 20, 30, 40, 50],
                              col4 = ["A", "B", "C", "D", "E"],
                              col5 = [100.0, 200.0, 300.0, 400.0, 500.0],
                              split = ["train", "train", "train", "valid", "valid"])
        td = TableDataset(testtable)
        @test td isa TableDataset{<:DataFrame}

        @test [data for data in td[1]] == [1, "a", 10, "A", 100.0, "train"]
        @test length(td) == 5
    end

    @testset "TableDataset from CSV" begin
        open("test.csv", "w") do io
            write(io, "col1,col2,col3,col4,col5, split\n1,a,10,A,100.,train")
        end
        testtable = CSV.File("test.csv")
        td = TableDataset(testtable)
        @test td isa TableDataset{<:CSV.File}
        @test [data for data in td[1]] == [1, "a", 10, "A", 100.0, "train"]
        @test length(td) == 1
        rm("test.csv")
    end
end
