function ulmfit_datadep_register()
    register(DataDep(
        "WikiText-103",
        """
        WikiText Long Term Dependency Language Modeling Dataset
        This is a language modelling dataset under Creative Commons Attribution-ShareAlike License. I contains over 100
        million tokens.
        """,
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
        post_fetch_method = function (file)
            unpack(file)
            dir = "wikitext-103"
            files = readdir(dir)
            mv.(joinpath.(dir, files), files)
            rm(dir)
            # some preprocessing
            corpus = read(open("wiki.train.tokens", "r"), String)
            corpus = split(lowercase(corpus), '\n')
            deleteat!(corpus, findall(x -> isequal(x, "")||isequal(x, " ")||(isequal(x[1:2], " =")&&isequal(x[prevind(x, lastindex(x), 1):end], "= ")), corpus))
            corpus = strip.(corpus)
            corpus = join(corpus, ' ')
            open("wiki.train.tokens", "w") do ftrain
                write(ftrain, corpus)
            end #do
        end # post_fetch_method
    ))

    #Pretrained ULMFiT language model weights
    register(DataDep(
        "Pretrained ULMFiT Language Model",
        """
        The pretrained Language Model weights trained on WikiText-103 corpus will be downloaded.
        These weights can be used to fine-tuning steps in ULMFiT classifier.
        This lanaguage model is originally trained by authors of ULMFiT paper Jeremy Howard and Sebastian Ruder.
        http://files.fast.ai/models/wt103/
        This contains the weights links for all original models by authors.
        """,
        "https://github.com/JuliaText/TextAnalysis.jl/releases/download/v0.6.0/ulmfit_lm_en.bson.tar.gz",
        "c71f580868fab7cc279cbceb0977ca26d252439a5cea604b2eebf6bf687127fa",
        post_fetch_method = function(fn)
            unpack(fn)
        end
    ))

    #Weights for ULMFiT binary sentiment classifier
    register(DataDep(
        "ULMFiT Sentiment Classifier",
        """
        Weights for the binary sentiment classifier, trained on IMDB movie review dataset,
        will be downloaded.
        """,
        "https://github.com/JuliaText/TextAnalysis.jl/releases/download/v0.6.0/ulmfit_sentiment_en.bson.tar.gz",# link
        post_fetch_method = function(fn)
            unpack(fn)
        end
    ))
end
