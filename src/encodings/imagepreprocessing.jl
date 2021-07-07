

"""
    ImagePreprocessing(C, T, stats)

Encodes `Image`s by converting them to a common color type `C`,
expanding the color channels and normalizing the channel values.

Encodes `Image{N}` -> `ImageTensor{N}` and decodes the reverse.
"""
struct ImagePreprocessing <: Encoding
    # intermediate color type to convert to
    C
    # number type of output image tensor
    T
end

function ImagePreprocessing(; C = RGB{N0f8}, T = Float32)

end

function encodedblock(ip::ImagePreprocessing, ::Image{N}) where N
    return ImageTensor{N}(colorchannels(ip.C))
end

decodedblock(::ImagePreprocessing, ::ImageTensor{N}) where N = Image{N}()

function encode(::ImagePreprocessing)
    # see imagepreprocessing.jl
end
