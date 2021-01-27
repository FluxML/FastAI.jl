
# from https://s3.amazonaws.com/fast-ai-imageclas/
const DATASETS = [
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
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
]


function init_datadeps()
    for (datasetname, checksum) in zip(FASTAI_DATASETS, CHECKSUMS)
        DataDeps.register(DataDep(
            "fastai-$datasetname",
            """
            $datasetname from the fastai dataset repository

            (https://s3.amazonaws.com/fast-ai-imageclas/)
            """,
            "https://s3.amazonaws.com/fast-ai-imageclas/$datasetname.tgz",
            checksum,
        ))
    end
end
