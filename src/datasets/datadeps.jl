
const DATASETS = [
    # Image classification
    # from https://s3.amazonaws.com/fast-ai-imageclas/
    "CUB_200_2011",
    "bedroom",
    "caltech_101",
    "cifar10",
    "cifar100",
    "food-101",
    "imagenette-160",
    "imagenette-320",
    "imagenette",
    "imagenette2-160",
    "imagenette2-320",
    "imagenette2",
    "imagewang-160",
    "imagewang-320",
    "imagewang",
    "imagewoof-160",
    "imagewoof-320",
    "imagewoof",
    "imagewoof2-160",
    "imagewoof2-320",
    "imagewoof2",
    "mnist_png",
    "mnist_var_size_tiny",
    "oxford-102-flowers",
    "oxford-iiit-pet",
    "stanford-cars"
]


const CHECKSUMS = [
    # Image classification
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "88daccb09b6fce93f45e6c09ddeb269cce705549e6bff322092a2a5a11489863",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "663c22f69c2802d85e2a67103c017e047096702ffddf9149a14011b7002539bf",
    "",
    "",
    "",
    "8a0f6ca04c2d31810dc08e739c7fa9b612e236383f70dd9fc6e5a62e672e2283",
    "",
    "",
    ""
]


function init_datadeps()
    for (datasetname, checksum) in zip(DATASETS, CHECKSUMS)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            $datasetname from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-imageclas/)
            """,
            "https://s3.amazonaws.com/fast-ai-imageclas/$datasetname.tgz",
            checksum,
            post_fetch_method = DataDeps.unpack,
        ))
    end
end
