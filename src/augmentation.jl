function augs_projection(; flipx = true, flipy = false)
    tfms = Identity()

    if flipx
        tfms = tfms |> Maybe(FlipX())
    end

    if flipy
        tfms = tfms |> Maybe(FlipY())
    end

    @show tfms
    return Maybe(FlipX())
end
