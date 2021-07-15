function accuracy_thresh(ŷs, ys, thresh = 0.5)
    mean((sigmoid.(ŷs) .> thresh) .== ys)
end
