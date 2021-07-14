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
        name, subfolder, checksum=""; extension="tgz", description="", datadepname=name, size="???")
    return FastAIDataset(name, subfolder, extension, description, checksum, datadepname, size)
end


const DATASETCONFIGS = [
    # imageclas
    FastAIDataset("CUB_200_2011", "imageclas", "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081", size = "1GiB"),
    FastAIDataset("bedroom", "imageclas", size="4.25GiB"),
    FastAIDataset("caltech_101", "imageclas"),
    FastAIDataset("cifar10", "imageclas", "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"),
    FastAIDataset("cifar100", "imageclas", "085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28"),
    FastAIDataset("food-101", "imageclas"),
    FastAIDataset("imagenette-160", "imageclas", "1bd650bc16884ca88e4f0f537ed8569b1f8d7ae865d37eba8ecdd87d9cd9dcfa", size="1.45GiB"),
    FastAIDataset("imagenette-320", "imageclas"),
    FastAIDataset("imagenette", "imageclas"),
    FastAIDataset("imagenette2-160", "imageclas", "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"),
    FastAIDataset("imagenette2-320", "imageclas", "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"),
    FastAIDataset("imagenette2", "imageclas", "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"),
    FastAIDataset("imagewang-160", "imageclas", "a0d360f9d8159055b3bf2b8926a51d19b2f1ff98a1eef6034e4b891c59ca3f1a", size="182MiB"),
    FastAIDataset("imagewang-320", "imageclas", "fd53301c335aa46f0f4add68dd471cd0b8b66412382cc36f5f510d0a03fb4d9d", size="639MiB"),
    FastAIDataset("imagewang", "imageclas"),
    FastAIDataset("imagewoof-160", "imageclas", "a0d360f9d8159055b3bf2b8926a51d19b2f1ff98a1eef6034e4b891c59ca3f1a"),
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
    FastAIDataset("biwi_head_pose", "imagelocal", "9cfefd53ed85f824c5908bc6eb21fc719583eec57a7df1d8141d3156645693cf", size="430MiB"),
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
    FastAIDataset("dogscats", "sample", "b79c0a5e4aa9ba7a0b83abbf61908c61e15bed0e5b236e86a0c4a080c8f70d7c", size="800MiB"),
    FastAIDataset("human_numbers", "sample"),
    FastAIDataset("imdb_sample", "sample"),
    FastAIDataset("mnist_sample", "sample"),
    FastAIDataset("mnist_tiny", "sample"),
    FastAIDataset("movie_lens_sample", "sample"),
    FastAIDataset("planet_sample", "sample"),
    FastAIDataset("planet_tiny", "sample"),

    # coco
    FastAIDataset("coco_sample", "coco", "56960c0ac09ff35cd8588823d37e1ed0954cb88b8bfbd214a7763e72f982911c", size="3GB"),
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
        post_fetch_method=DataDeps.unpack,
    )
end

function initdatadeps()
    for d in DATASETCONFIGS
        DataDeps.register(DataDep(d))
    end
end
