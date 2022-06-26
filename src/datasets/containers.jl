
function _ts2df(
    full_file_path_and_name:: String,
    replace_missing_vals_with:: String="NaN", # The value that missing values in the text file should be replaced with prior to parsing.
)
    "Load data from a .ts file into a DataFrames.jl DataFrame"

    # Initialize flags and variables used when parsing the file
    metadata_started = false 
    data_started = false

    has_problem_name_tag = false
    has_timestamps_tag = false
    has_univariate_tag = false
    has_class_labels_tag = false
    has_data_tag = false

    previous_timestamp_was_int = nothing
    prev_timestamp_was_timestamp = nothing
    num_dimensions = nothing
    is_first_case = true
    instance_list = []
    class_val_list = []
    line_num = 0
    series_length = 0

    timestamps = false
    class_labels = false

    open(full_file_path_and_name, "r") do file
        for ln in eachline(file)
            # Strip white space from start/end of line and change to
            # lowercase for use below
            ln = lowercase(strip(ln, ' '))
            # Empty lines are valid at any point in a file
            if !isempty(ln)
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if startswith(ln, "@problemname")
                    # Check that the associated value is valid
                    tokens = split(ln, " ")
                    token_len = length(tokens)

                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = true
                    metadata_started = true

                elseif startswith(ln, "@timestamps")
                    # Check that the associated value is valid
                    tokens = split(ln, " ")
                    token_len = length(tokens)

                    if tokens[2] == "true"
                        timestamps = true
                    else
                        timestamps = false
                    end

                    has_timestamps_tag = true
                    metadata_started = true

                elseif startswith(ln, "@univariate")
                    # Check that the associated value is valid
                    tokens = split(ln, " ")
                    token_len = length(tokens)

                    if tokens[2] == "true"
                        # univariate = true
                    else
                        # univariate = false
                    end

                    has_univariate_tag = true
                    metadata_started = true

                elseif startswith(ln, "@serieslength")
                    # Check that the associated value is valid
                    tokens = split(ln, " ")
                    token_len = length(tokens)

                    series_length = parse(Int, tokens[2])

                elseif startswith(ln, "@classlabel")
                    # Check that the associated value is valid
                    tokens = split(ln, " ")
                    token_len = length(tokens)

                    if tokens[2] == "true"
                        class_labels = true
                    else
                        class_labels = false
                    end

                    has_class_labels_tag = true
                    class_label_list = [strip(token, ' ') for token in tokens[3:end]]
                    metadata_started = true
                
                # Check if this line contains the start of data
                elseif startswith(ln, "@data")

                    if ln != "@data"
                    end

                    if data_started == true && metadata_started == false
                    else
                        has_data_tag = true
                        data_started = true
                    end
                
                elseif data_started
                    
                    # Check that a full set of metadata has been provided

                    if (!has_problem_name_tag || !has_timestamps_tag || !has_univariate_tag || !has_class_labels_tag || !has_data_tag)
                    end

                    # Replace any missing values with the value specified

                    ln = replace(ln, '?' => replace_missing_vals_with)
                    
                    # Check if we dealing with data that has timestamps

                    if timestamps
                        #! Need To Add Code.
                    else
                        dimensions = split(ln, ':')

                        # If first row then note the number of dimensions (that must be the same for all cases)

                        if is_first_case
                            num_dimensions = length(dimensions)

                            if class_labels
                                num_dimensions -= 1
                            end
                            
                            # for _dim in 1:num_dimensions
                            #     push!(instance_list, [])
                            # end

                            is_first_case = false
                        end
                        
                        # See how many dimensions that the case whose data
                        # in represented in this line has

                        this_line_num_dim = length(dimensions)

                        if class_labels
                            this_line_num_dim -= 1
                        end

                        # All dimensions should be included for all series,
                        # even if they are empty

                        if this_line_num_dim != num_dimensions
                        end

                        # Process the data for each dimension

                        arr = Array{Float32, 2}(undef, num_dimensions, series_length)

                        for dim in 1:num_dimensions
                            dimension = strip(dimensions[dim], ',')

                            if !isempty(dimension)
                                data_series = split(dimension, ',')
                                data_series = [parse(Float32, i) for i in data_series]
                                arr[dim, 1:end] = data_series
                                # println(data_series)
                                # data_series = [parse(Float32, i) for i in data_series]
                                # push!(instance_list[dim], data_series)
                            else
                                tmp = Array{Float32, 1}(undef, 100)
                                arr[dim, 1:end] = tmp
                                # push!(instance_list[dim], [])
                            end
                        end

                        push!(instance_list, arr)

                        push!(class_val_list, strip(dimensions[num_dimensions+1], ' '))

                    end
                end
            end
            line_num += 1
        end
    end

    if line_num > 0

        num_samples = length(instance_list)
        series_length = size(instance_list[1])[2]

        data = Array{Float32, 3}(undef, num_samples, num_dimensions, series_length)

        for sample in 1:num_samples
            data[sample, 1:end, 1:end] = instance_list[sample]
        end

        # Check if we should return any associated class labels separately
        if class_labels
            return data, class_val_list
        else
            return data
        end

    end

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
    elseif endswith(file, ".txt")
        return read(file, String)
    elseif endswith(file, ".ts")
        return _ts2df(file)
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

function Base.getindex(dataset::FastAI.Datasets.TableDataset, idx)
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

function Base.length(dataset::TableDataset)
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

Base.getindex(dataset::TableDataset{<:DataFrame}, idx) = dataset.table[idx, :]
Base.length(dataset::TableDataset{<:DataFrame}) = nrow(dataset.table)

Base.getindex(dataset::TableDataset{<:CSV.File}, idx) = dataset.table[idx]
Base.length(dataset::TableDataset{<:CSV.File}) = length(dataset.table)


#TimeSeriesDataset

struct TimeSeriesDataset{T<:AbstractArray}
    table::T
end

function TimeSeriesDataset(path::Union{AbstractPath, String})
    data = loadfile(string(path))
    TimeSeriesDataset(data[1])
end

function Base.getindex(dataset::TimeSeriesDataset, idx)
    dataset.table[idx,:,:]
end

function Base.length(dataset::TimeSeriesDataset)
    size(dataset.table)[1] 
end

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

#! How to include the file.
# @testset "TimeSeriesDataset" begin
#     @testset "TimeSeriesDataset from TS" begin
#         # Size 159 KB
#         tsd = TimeSeriesDataset("/Users/saksham/Downloads/AtrialFibrillation/AtrialFibrillation_TRAIN.ts") 
#         @test tsd isa TimeSeriesDataset{}
#         @test size(getindex(tsd, 10)) == (2, 640)
#         @test length(tsd) ==15
#     end
# end

# @testset "TimeSeriesDataset" begin
#     @testset "TimeSeriesDataset from TS" begin
#         temp = mktempdir()
#         downpath = joinpath(temp, "temp.zip")
#         path = Downloads.download("http://timeseriesclassification.com/Downloads/AtrialFibrillation.zip", downpath)
#         InfoZIP.unzip(path, temp)
#         tsd = TimeSeriesDataset(joinpath(temp, "AtrialFibrillation_TRAIN.ts"))
#         @test tsd isa TimeSeriesDataset{}
#         @test size(getindex(tsd, 10)) == (2, 640)
#         @test length(tsd) ==15
#     end
# end