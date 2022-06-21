const RE_IMAGEFILE = r".*\.(gif|jpe?g|tiff?|png|webp|bmp)$"i
isimagefile(f) = matches(RE_IMAGEFILE, f)

function maskfromimage(a::AbstractArray{<:Gray{T}}, classes) where {T}
    maskfromimage(reinterpret(T, a), classes)
end
function maskfromimage(a::AbstractArray{<:Normed{T}}, classes) where {T}
    maskfromimage(reinterpret(T, a), classes)
end
function maskfromimage(a::AbstractArray{I}, classes) where {I <: Integer}
    a .+= one(I)
    return IndirectArray(a, classes)
end

"""
    loadmask(file, classes)

Load a segmentation mask from an image file. Returns an efficiently stored
array of type `eltype(classes)`.

"""
loadmask(file, classes) = maskfromimage(loadfile(file), classes)
