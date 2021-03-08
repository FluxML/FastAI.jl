struct FastAIDataset
    name
    subfolder
    extension
    description
    checksum
    datadepname
    size
end

const ROOT_URL = "https://s3.amazonaws.com/fast-ai-"

function FastAIDataset(
        name, subfolder, checksum = ""; extension = "tgz", description = "", datadepname = name, size = "???")
    return FastAIDataset(name, subfolder, extension, description, checksum, datadepname, size)
end


const DATASETCONFIGS = [
    # imageclas
    FastAIDataset("CUB_200_2011", "imageclas"),
    FastAIDataset("bedroom", "imageclas"),
    FastAIDataset("caltech_101", "imageclas"),
    FastAIDataset("cifar10", "imageclas", "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"),
    FastAIDataset("cifar100", "imageclas", "085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28"),
    FastAIDataset("food-101", "imageclas"),
    FastAIDataset("imagenette-160", "imageclas"),
    FastAIDataset("imagenette-320", "imageclas"),
    FastAIDataset("imagenette", "imageclas"),
    FastAIDataset("imagenette2-160", "imageclas", "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"),
    FastAIDataset("imagenette2-320", "imageclas", "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"),
    FastAIDataset("imagenette2", "imageclas"),
    FastAIDataset("imagewang-160", "imageclas"),
    FastAIDataset("imagewang-320", "imageclas"),
    FastAIDataset("imagewang", "imageclas"),
    FastAIDataset("imagewoof-160", "imageclas"),
    FastAIDataset("imagewoof-320", "imageclas"),
    FastAIDataset("imagewoof", "imageclas"),
    FastAIDataset("imagewoof2-160", "imageclas", "663c22f69c2802d85e2a67103c017e047096702ffddf9149a14011b7002539bf"),
    FastAIDataset("imagewoof2-320", "imageclas"),
    FastAIDataset("imagewoof2", "imageclas"),
    FastAIDataset("mnist_png", "imageclas", "9e18edaa3a08b065d8f80a019ca04329e6d9b3e391363414a9bd1ada30563672"),
    FastAIDataset("mnist_var_size_tiny", "imageclas", "8a0f6ca04c2d31810dc08e739c7fa9b612e236383f70dd9fc6e5a62e672e2283"),
    FastAIDataset("oxford-102-flowers", "imageclas"),
    FastAIDataset("oxford-iiit-pet", "imageclas"),
    FastAIDataset("stanford-cars", "imageclas"),

    # nlp
    FastAIDataset("ag_news_csv", "nlp"),
    FastAIDataset("amazon_review_full_csv", "nlp"),
    FastAIDataset("amazon_review_polarity_csv", "nlp"),
    FastAIDataset("dbpedia_csv", "nlp"),
    FastAIDataset("giga-fren", "nlp"),
    FastAIDataset("imdb", "nlp"),
    FastAIDataset("sogou_news_csv", "nlp"),
    FastAIDataset("wikitext-103", "nlp"),
    FastAIDataset("wikitext-2", "nlp"),
    FastAIDataset("yahoo_answers_csv", "nlp"),
    FastAIDataset("yelp_review_full_csv", "nlp"),
    FastAIDataset("yelp_review_polarity_csv", "nlp"),

    # imagelocal
    FastAIDataset("biwi_head_pose", "imagelocal"),
    FastAIDataset("camvid", "imagelocal"),
    FastAIDataset("pascal-voc", "imagelocal"),
    FastAIDataset("pascal_2007", "imagelocal"),
    FastAIDataset("pascal_2012", "imagelocal"),
    FastAIDataset("siim_small", "imagelocal"),
    FastAIDataset("skin-lesion", "imagelocal"),
    FastAIDataset("tcga-small", "imagelocal"),

    # sample
    FastAIDataset("adult_sample", "sample"),
    FastAIDataset("biwi_sample", "sample"),
    FastAIDataset("camvid_tiny", "sample", "cd42a9bdd8ad3e0ce87179749beae05b4beb1ae6ab665841180b1d8022fc230b"),
    FastAIDataset("dogscats", "sample"),
    FastAIDataset("human_numbers", "sample"),
    FastAIDataset("imdb_sample", "sample"),
    FastAIDataset("mnist_sample", "sample"),
    FastAIDataset("mnist_tiny", "sample"),
    FastAIDataset("movie_lens_sample", "sample"),
    FastAIDataset("planet_sample", "sample"),
    FastAIDataset("planet_tiny", "sample"),

    # coco
    FastAIDataset("coco_sample", "coco", "56960c0ac09ff35cd8588823d37e1ed0954cb88b8bfbd214a7763e72f982911c", size = "3GB"),
    FastAIDataset("train2017", "coco", datadepname="coco-train2017", extension="zip"),
    FastAIDataset("val2017", "coco", datadepname="coco-val2017", extension="zip"),
    FastAIDataset("test2017", "coco", datadepname="coco-test2017", extension="zip"),
    FastAIDataset("unlabeled2017", "coco", datadepname="coco-unlabeled2017", extension="zip"),
    FastAIDataset("image_info_test2017", "coco", datadepname="coco-image_info_test2017", extension="zip"),
    FastAIDataset("image_info_unlabeled2017", "coco", datadepname="coco-image_info_unlabeled2017", extension="zip"),
    FastAIDataset("annotations_trainval2017", "coco", datadepname="coco-annotations_trainval2017", extension="zip"),
    FastAIDataset("stuff_annotations_trainval2017", "coco", datadepname="coco-stuff_annotations_trainval2017", extension="zip"),
    FastAIDataset("panoptic_annotations_trainval2017", "coco", datadepname="coco-panoptic_annotations_trainval2017", extension="zip"),
]

const DATASETS = [d.datadepname for d in DATASETCONFIGS]
const DATASETS_IMAGECLASSIFICATION = vcat(
    [d.datadepname for d in DATASETCONFIGS if d.subfolder == "imageclas"],
    ["mnist_sample", "mnist_tiny", "dogscats"],

)


function DataDeps.DataDep(d::FastAIDataset)
    return DataDep(
        "fastai-$(d.datadepname)",
        """
        "$(d.name)" from the fastai dataset repository (https://course.fast.ai/datasets)

        $(d.description)

        Download size: $(d.size)
        """,
        "$(ROOT_URL)$(d.subfolder)/$(d.name).$(d.extension)",
        d.checksum,
        post_fetch_method = DataDeps.unpack,
    )
end

function initdatadeps()
    for d in DATASETCONFIGS
        DataDeps.register(DataDep(d))
    end
end
