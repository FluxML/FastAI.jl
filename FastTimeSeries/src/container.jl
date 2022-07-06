#= TODO: loadfile

    elseif endswith(file, ".ts")
        return _ts2df(file)

=#

Datasets.loadfile(file::String, ::Val{:ts}) = _ts2df(file)

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
