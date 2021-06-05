include("imports.jl")

@testset ExtendedTestSet "TableDataset" begin

    @testset ExtendedTestSet "TableDataset from rowaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = false
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = true
        
        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test [data for data in getobs(td, 1)] == [1, 4.0, "7"]
        @test nobs(td) == 3
    end

    @testset ExtendedTestSet "TableDataset from columnaccess table" begin
        Tables.columnaccess(::Type{<:Tables.MatrixTable}) = true
        Tables.rowaccess(::Type{<:Tables.MatrixTable}) = false
        
        testtable = Tables.table([1 4.0 "7"; 2 5.0 "8"; 3 6.0 "9"])
        td = TableDataset(testtable)

        @test getobs(td, 2) == [2, 5.0, "8"]
        @test nobs(td) == 3
    end

    @testset ExtendedTestSet "TableDataset from DataFrames" begin
        testtable = DataFrame(
            col1=[1, 2, 3, 4, 5], 
            col2=["a", "b", "c", "d", "e"], 
            col3=[10, 20, 30, 40, 50], 
            col4=["A", "B", "C", "D", "E"],
            col5=[100., 200., 300., 400., 500.],
            split=["train", "train", "train", "valid", "valid"]
        )
        @test TableDataset(testtable) isa TableDataset{<:DataFrame}
        td = TableDataset(testtable)

        @test [data for data in getobs(td, 1)] == [1, "a", 10, "A", 100., "train"]
        @test nobs(td) == 5 
    end

    @testset ExtendedTestSet "TableDataset from CSV" begin
        path = datasetpath("adult_sample")
        testtable = CSV.File(joinpath(path, "adult.csv"))
        @test TableDataset(testtable) isa TableDataset{<:CSV.File}
        td = TableDataset(testtable)

        @test [data for data in getobs(td, 2)] == [44, " Private", 236746, " Masters", 14.0, " Divorced", " Exec-managerial", " Not-in-family", " White", " Male", 10520, 0, 45, " United-States", ">=50k"]
        @test nobs(td) == 32561
        
    end
end