
const DATASETS = vcat(
    DATASETS_IMAGECLASSIFICATION,
    DATASETS_IMAGELOCALIZATION,
    DATASETS_NLP,
    DATASETS_SAMPLE,
)

const CHECKSUMS = vcat(
    CHECKSUMS_IMAGECLASSIFICATION,
    CHECKSUMS_IMAGELOCALIZATION,
    CHECKSUMS_NLP,
    CHECKSUMS_SAMPLE,
)

function init_datadeps()
    # Image classification datasets
    for (datasetname, checksum) in zip(DATASETS_IMAGECLASSIFICATION, CHECKSUMS_IMAGECLASSIFICATION)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            "$datasetname" from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-imageclas/)
            """,
            "https://s3.amazonaws.com/fast-ai-imageclas/$datasetname.tgz",
            checksum,
            post_fetch_method = DataDeps.unpack,
        ))
    end

    # Image localization datasets
    for (datasetname, checksum) in zip(DATASETS_IMAGELOCALIZATION, CHECKSUMS_IMAGELOCALIZATION)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            "$datasetname" from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-imagelocal/)
            """,
            "https://s3.amazonaws.com/fast-ai-imagelocal/$datasetname.tgz",
            checksum,
            post_fetch_method = DataDeps.unpack,
        ))
    end

    # NLP datasets
    for (datasetname, checksum) in zip(DATASETS_NLP, CHECKSUMS_NLP)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            "$datasetname" from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-nlp/)
            """,
            "https://s3.amazonaws.com/fast-ai-nlp/$datasetname.tgz",
            checksum,
            post_fetch_method = DataDeps.unpack,
        ))
    end

    # Sample datasets
    for (datasetname, checksum) in zip(DATASETS_SAMPLE, CHECKSUMS_SAMPLE)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            "$datasetname" from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-sample/)
            """,
            "https://s3.amazonaws.com/fast-ai-sample/$datasetname.tgz",
            checksum,
            post_fetch_method = DataDeps.unpack,
        ))
    end
end
