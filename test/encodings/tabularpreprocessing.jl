include("../imports.jl")


@testset "TabularPreprocessing" begin
    cols = [:col1, :col2, :col3, :col4, :col5]
    vals = [1, 2, 3, "a", "x"]
    row = NamedTuple(zip(cols, vals))

    catcols = (:col4, :col5)
    contcols = (:col1, :col2, :col3)

    col1_mean, col1_std = 10, 100
    col2_mean, col2_std = 100, 10
    col3_mean, col3_std = 15, 1

    normdict = Dict(
        :col1 => (col1_mean, col1_std),
        :col2 => (col2_mean, col2_std),
        :col3 => (col3_mean, col3_std)
    )

    tfm = TabularPreprocessing(
        NormalizeRow(normdict, contcols)
    )

    block = TableRow(
        catcols,
        contcols,
        Dict(:col4=>["a", "b"], :col5=>["x", "y", "z"])
    )

    testencoding(tfm, block, row)

    @testset ExtendedTestSet "" begin
        testencoding(setup(TabularPreprocessing, block, TableDataset(DataFrame([row, row]))), block, row)
    end
end
