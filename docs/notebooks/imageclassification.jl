using FastAI

DATASETNAME = "imagenette2-160";

##

taskdata = Datasets.loadtaskdata(Datasets.datasetpath(DATASETNAME), ImageClassificationTask)
image, class = getobs(taskdata, 1)
