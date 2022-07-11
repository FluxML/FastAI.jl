function TextClassficationSingle(blocks::Tuple{<:Paragraph, <:Label}, data=nothing)
    return SupervisedTask(
        blocks,
        (
            
            OneHot()
        )
    )
end