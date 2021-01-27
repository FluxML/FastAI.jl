
include("../imports.jl")


@testset ExtendedTestSet "Data container transformations" begin
    @testset ExtendedTestSet "mapdata" begin
        data = 1:10
        mdata = mapdata(-, data)
        @test getobs(mdata, 8) == -8

        mdata2 = mapdata((-, x -> 2x), data)
        @test getobs(mdata2, 8) == (-8, 16)
    end

    @testset ExtendedTestSet "filterdata" begin
        data = 1:10
        fdata = filterdata(>(5), data)
        @test nobs(fdata) == 5
    end

    @testset ExtendedTestSet "splitdata" begin
        data = -10:10
        datas = splitdata(>(0), data)
        length(datas) == 2
    end
end
