const WEIGHTS = [
    "resnet50",
    "densenet121",
    "googlenet",
    "squeezenet",
    "vgg19",
]

const CHECKSUMS = [
    "88e9196c1d451186139dbc93d538092e21824978c5ec647a6599ad4152e79870",
    "",
    "",
    "",
    "",
]

function initdatadeps()
    for (name, checksum) in zip(WEIGHTS, CHECKSUMS)

        DataDeps.register(DataDep(
            "weights-$name",
            """
            Pretrained weights for model "$name".
            """,
            "https://github.com/darsnack/MetalheadWeights/raw/main/$name.bson",
            checksum,
        ))
    end
end
