struct FastAIDataset
    name
    subfolder
    extension
    description
    checksum
    datadepname
    subpath
    size
end

struct TSClassificationDataset
    name
    extension
    description
    checksum
    datadepname
    size 
end

const ROOT_URL_FastAI = "https://s3.amazonaws.com/fast-ai-"
const ROOT_URL_TSClassification = "http://www.timeseriesclassification.com/Downloads"

function FastAIDataset(
        name, subfolder, checksum="";
        extension="tgz",
        description="",
        datadepname=name,
        subpath=name,
        size="???")
    return FastAIDataset(name, subfolder, extension, description, checksum, datadepname, subpath, size)
end

function TSClassificationDataset(
        name, checksum="";
        extension="zip",
        description="",
        datadepname="",
        size="???")
    return TSClassificationDataset(name, extension, description, checksum, datadepname, size)
end

const DESCRIPTIONS = Dict(
    "imagenette" => "A subset of 10 easily classified classes from Imagenet: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute",
    "imagewoof" => "A subset of 10 harder to classify classes from Imagenet (all dog breeds): Australian terrier, Border terrier, Samoyed, beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, dingo, golden retriever, Old English sheepdog",
    "food-101" => "101 food categories, with 101,000 images; 250 test images and 750 training images per class. The training images were not cleaned. All images were rescaled to have a maximum side length of 512 pixels.",
    "ECG5000" => "The original dataset for \"ECG5000\" is a 20-hour long ECG downloaded from Physionet. The name is BIDMC Congestive Heart Failure Database(chfdb) and it is record \"chf07\".",
    "AtrialFibrillation" => "This is a physionet dataset of two-channel ECG recordings has been created from data used in the Computers in Cardiology Challenge 2004, an open competition with the goal of developing automated methods for predicting spontaneous termination of atrial fibrillation (AF).",
)

const DATASETCONFIGS = [
    # imageclas
    FastAIDataset("CUB_200_2011", "imageclas", "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081", size="1GiB"),
    FastAIDataset("bedroom", "imageclas", "7c95250ccb177c582f602c08f239c71f7a70512729d2e078925261cf5e349f5d", size="4.25GiB"),
    FastAIDataset("caltech_101", "imageclas", "af6ece2f339791ca20f855943d8b55dd60892c0a25105fcd631ee3d6430f9926", size="126MiB", subpath="101_ObjectCategories"),
    FastAIDataset("cifar10", "imageclas", "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"),
    FastAIDataset("cifar100", "imageclas", "085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28"),
    FastAIDataset("food-101", "imageclas", "abc3d6b03a9886fdea6d2a124cf88e22a99dfdb03085b2478be97de3f8e4679f", size="5.3GB", description=DESCRIPTIONS["food-101"]),
    FastAIDataset("imagenette-160", "imageclas", "1bd650bc16884ca88e4f0f537ed8569b1f8d7ae865d37eba8ecdd87d9cd9dcfa", size="1.45GiB", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagenette-320", "imageclas", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagenette", "imageclas", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagenette2-160", "imageclas", "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagenette2-320", "imageclas", "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagenette2", "imageclas", "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd", description=DESCRIPTIONS["imagenette"]),
    FastAIDataset("imagewang-160", "imageclas", "a0d360f9d8159055b3bf2b8926a51d19b2f1ff98a1eef6034e4b891c59ca3f1a", size="182MiB"),
    FastAIDataset("imagewang-320", "imageclas", "fd53301c335aa46f0f4add68dd471cd0b8b66412382cc36f5f510d0a03fb4d9d", size="639MiB"),
    FastAIDataset("imagewang", "imageclas"),
    FastAIDataset("imagewoof-160", "imageclas", "a0d360f9d8159055b3bf2b8926a51d19b2f1ff98a1eef6034e4b891c59ca3f1a", description=DESCRIPTIONS["imagewoof"]),
    FastAIDataset("imagewoof-320", "imageclas", description=DESCRIPTIONS["imagewoof"]),
    FastAIDataset("imagewoof", "imageclas", description=DESCRIPTIONS["imagewoof"]),
    FastAIDataset("imagewoof2-160", "imageclas", "b5ffa16037e07f60882434f55b7814a3d44483f2a484129f251604bc0d0f8172", description=DESCRIPTIONS["imagewoof"]),
    FastAIDataset("imagewoof2-320", "imageclas", "7db6120fdb9ae079e26346f89e7b00d7f184f8137791609b97fd0405d3f92305", description=DESCRIPTIONS["imagewoof"], size="313MB"),
    FastAIDataset("imagewoof2", "imageclas", "de3f58c4ea3e042cf3f8365fbc699288cfe1d8c151059040d181c221bd5a55b8", description=DESCRIPTIONS["imagewoof"], size="1.25GiB"),
    FastAIDataset("mnist_png", "imageclas", "9e18edaa3a08b065d8f80a019ca04329e6d9b3e391363414a9bd1ada30563672"),
    FastAIDataset("mnist_var_size_tiny", "imageclas", "8a0f6ca04c2d31810dc08e739c7fa9b612e236383f70dd9fc6e5a62e672e2283"),
    FastAIDataset("oxford-102-flowers", "imageclas"),
    FastAIDataset("oxford-iiit-pet", "imageclas"),
    FastAIDataset("stanford-cars", "imageclas"),

    # nlp
    FastAIDataset("ag_news_csv", "nlp", "9a8c300eabb45750237fcc669f61cb8a3448f3ef6f6098e1ce340e444f6872be", size="11MB"),
    FastAIDataset("amazon_review_full_csv", "nlp", "4af62eeee139d0142e0747340b68646d23483d9475c33ea0641ee9175b423443", size="600MB"),
    FastAIDataset("amazon_review_polarity_csv", "nlp", "d2a3ee7a214497a5d1b8eaed7c8d7ba2737de00ada3b0ec46243983efa100361", size="600MB"),
    FastAIDataset("dbpedia_csv", "nlp", "42db5221ddedddb673a4cabcc5f3a7d869714c878bcfe4ba94b29d14aa38e417", size="65MB"),
    FastAIDataset("giga-fren", "nlp", "11c97af99471fe641f210d8b86ccccf3b298b9199853987ee53892d709d7ca6b", size="2.4GB"),
    FastAIDataset("imdb", "nlp", "d501018afa17aee9fa1ebe8ac29859a5609980e13dc6e611aa21567cc357351f", size="140MB"),
    FastAIDataset("sogou_news_csv", "nlp", "6b77fc935561d339b82aa552d7e31ea59eff492a494920579b3ce70604efb5c2", size="360MB"),
    FastAIDataset("wikitext-103", "nlp", "27b89e94d98a9f9db74588a2e75b04378ee21569ce55d329d3e73e27d0952551", size="181MB"),
    FastAIDataset("wikitext-2", "nlp", "4e39df0e84453ae2f3d34333de2a9d8e57560a7a6e621f13e11dc21241320074", size="4MB"),
    FastAIDataset("yahoo_answers_csv", "nlp", "2d4277855faf8b35259009425fa8f7fe1888b5644b47165508942d000f4c96ae", size="305MB"),
    FastAIDataset("yelp_review_full_csv", "nlp", "56006b0a17a370f1e366504b1f2c3e3754e4a3dda17d3e718a885c552869a559", size="187MB"),
    FastAIDataset("yelp_review_polarity_csv", "nlp", "528f22e286cad085948acbc3bea7e58188416546b0e364d0ae4ca0ce666abe35", size="158MB"),

    # imagelocal
    FastAIDataset("biwi_head_pose", "imagelocal", "9cfefd53ed85f824c5908bc6eb21fc719583eec57a7df1d8141d3156645693cf", size="430MiB"),
    FastAIDataset("camvid", "imagelocal", "11db05fc3ee727fb17de7499380b20258a41beeb1002a2aee2c2244a472a4a45", size="571MB"),
    FastAIDataset("pascal-voc", "imagelocal", "10fc13a659da20fdd8302dd394d88ca7e4e60e69fd8a5212c3e3357964a58215", size="4.3GB"),
    FastAIDataset("pascal_2007", "imagelocal"),
    FastAIDataset("pascal_2012", "imagelocal"),
    FastAIDataset("siim_small", "imagelocal"),
    FastAIDataset("skin-lesion", "imagelocal"),
    FastAIDataset("tcga-small", "imagelocal"),

    # sample
    FastAIDataset("adult_sample", "sample", "47ecd1848abc976643ee82d8788b712e3006d629bbc7554efa1077a91579e99e", size="3.8MB"),
    FastAIDataset("biwi_sample", "sample"),
    FastAIDataset("camvid_tiny", "sample", "cd42a9bdd8ad3e0ce87179749beae05b4beb1ae6ab665841180b1d8022fc230b"),
    FastAIDataset("dogscats", "sample", "b79c0a5e4aa9ba7a0b83abbf61908c61e15bed0e5b236e86a0c4a080c8f70d7c", size="800MiB"),
    FastAIDataset("human_numbers", "sample"),
    FastAIDataset("imdb_sample", "sample", "8e776d995296136b3f9a3cf001796d886cb0b60e86877ce71c7abbdc3c247341", size="4KB"),
    FastAIDataset("mnist_sample", "sample", "b373a14f282298aeba0f7dd56b7cdb6c2401063d4f118c39c54982907760bd38", size="3MB"),
    FastAIDataset("mnist_tiny", "sample", "0d1fedf86243931aa3fc065d2cf4ffab339a972958d8594ae993ee32bd8e15b9", size="300KB"),
    FastAIDataset("movie_lens_sample", "sample"),
    FastAIDataset("planet_sample", "sample", "f2509212bb2dcdc147423b164564f2e63cae1d1db0b504166e5b92cfbcbb3b4c", size="14.8MB"),
    FastAIDataset("planet_tiny", "sample", "41a5fdd82db1c9fb2cff17e1a1270102414a25a34b21b770f953d28483961edb", size="1MB"),

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

    # timeseries
    TSClassificationDataset("ECG5000", "41f6de20ac895e9ce31753860995518951f1ed42a405d0e51c909d27e3b3c5a4", description = DESCRIPTIONS["ECG5000"] ,datadepname="ecg5000", size="10MB" ),
    TSClassificationDataset("AtrialFibrillation", "218abad67d58190a6daa1a27f4bd58ace6e18f80fb59fb2c7385f0d2d4b411a2", description = DESCRIPTIONS["AtrialFibrillation"], datadepname = "atrial", size = "226KB"),
    
]

const DATASETS = [d.datadepname for d in DATASETCONFIGS]
const DATASETS_IMAGECLASSIFICATION = vcat(
    [d.datadepname for d in DATASETCONFIGS if ((typeof(d) == FastAIDataset) &&  d.subfolder == "imageclas")],
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
        "$(ROOT_URL_FastAI)$(d.subfolder)/$(d.name).$(d.extension)",
        d.checksum,
        post_fetch_method=function (f)
            DataDeps.unpack(f)
            extracted = readdir(pwd())[1]
            temp = mktempdir()
            mv(extracted, temp, force=true)
            mv(temp, pwd(), force=true)
        end,
    )
end

function DataDeps.DataDep(d::TSClassificationDataset)
    return DataDep(
        "fastai-$(d.datadepname)",
        """
        "$(d.name)" from the UEA and UCR time reries classification repository (http://timeseriesclassification.com/)

        $(d.description)

        Download size: $(d.size)
        """,
        "$(ROOT_URL_TSClassification)/$(d.name).$(d.extension)",
        d.checksum,
        post_fetch_method=function (f)
            DataDeps.unpack(f)
        end,
    )
end

function initdatadeps()
    for d in DATASETCONFIGS
        DataDeps.register(DataDep(d))
    end
end