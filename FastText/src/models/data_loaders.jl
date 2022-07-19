using Random: shuffle!

"""
Default Data Loaders for ULMFiT training for Sentiment Analysis
 - WikiText-103 corpus is to pre-train the Language model
 - IMDB movie review dataset - unsup data is used for fine-tuning Language Mode for Sentiment Analysis
 - IMDB movie review dataset - labelled data is used for training classifier for Sentiment Analysis
"""

# WikiText-103 corpus loade

# This funciton given as an example of doing preprocessing for such corpus.
# In this IMDB corpus, it was noticed that after tokenization some tokens were mixture of
# some valid words and punctuations like ',', '.' etc which leads to increase in UNKNOWN token count in the corpus,
# to reduce the UNKNOWN count those tokens are splitted using `split_worda` and `put` functions.
function imdb_preprocess(doc::AbstractDocument)
    ## Edit here if any preprocessing step is needed ##
    function put(en, symbol)
        l = length(en)
        (l == 1) && return en
        for i=1:l-1
            insert!(en, i*2, string(symbol))
        end
        return en
    end
    function split_word(word, symbol)
        length(word) == 1 && return [word]
        return split(word, symbol)
    end
    text_ = doc
    remove_corrupt_utf8!(text_)
    remove_case!(text_)
    prepare!(text_, strip_html_tags)
    tokens_ = tokens(text_)
    for symbol in [',', '.', '-', '/', "'s"]
        tokens_ = split_word.(tokens_, symbol)
        temp = []
        for token_ in tokens_
            try
                append!(temp, put(token_, symbol))
            catch
                append!(temp, token_)
            end
        end
        tokens_ = temp
    end
    deleteat!(tokens_, findall(x -> isequal(x, "")||isequal(x, " "), tokens_))
    return tokens_
end

# Loads WikiText-103 corpus and output a Channel to give a mini-batch at each call
function load_wikitext_103(batchsize::Integer=16, bptt::Integer=70; type = "train")
    corpuspath = joinpath(datadep"WikiText-103", "wiki.$(type).tokens")
    # corpus = read(open(corpuspath, "r"), String)
    # corpus = tokenize(corpus)
    corpus = readlines("/home/luna/Documents/wiki.txt")

    return Channel(x -> generator(x, corpus; batchsize = batchsize, bptt = bptt));
end

# IMDB Data loaders for Sentiment Analysis specifically
# IMDB data loader for fine-tuning Language Model
function imdb_fine_tune_data(batchsize::Integer=16, bptt::Integer=70, num_examples::Integer=50000)
    imdb_dataset = IMDB("train_unsup")
    dataset = []
    for path in imdb_dataset.filepaths[1:num_examples]   #extract data from the files in directory and put into channel
        open(path) do fileio
            cur_text = read(fileio, String)
            append!(dataset, imdb_preprocess(StringDocument(cur_text)))
        end #open
    end #for
    return Channel(x -> generator(x, dataset; batchsize=batchsize, bptt=bptt))
end

# IMDB data loader for training classifier
function imdb_classifier_data(batchsize::Integer)
    filepaths = IMDB("train_neg").filepaths
    append!(filepaths, IMDB("train_pos").filepaths)
    [shuffle!(filepaths) for _=1:10]
    corpus = Corpus(FileDocument.(filepaths))
    Channel(csize=2) do docs
        n_batches = Int(floor(length(corpus)/batchsize))
        put!(docs, n_batches)
        for i=1:n_batches
            X, Y = [], []
            for j=1:batchsize
                path = filepaths[(i-1)*batchsize+j]
                cur_text = text(corpus[(i-1)*batchsize+j])
                tokens = imdb_preprocess(StringDocument(cur_text))
                push!(X, tokens)
                y = (parse(Int32, split(path, '_')[2][1:end-4]) > 5) ? Flux.onehotbatch([1], 1:2) : Flux.onehotbatch([2], 1:2)
                push!(Y, y)
            end#for
            X = pre_pad_sequences(X, "_pad_")
            put!(docs, [Flux.batch(X[k][j] for k=1:batchsize) for j=1:length(X[1])])
            put!(docs, cat(Y..., dims=2))
        end #for
    end #channel
end
