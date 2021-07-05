struct TabularTransforms <: PipelineStep
	tfms
end

function run(tt::TabularTransforms, _, sample)
	DataAugmentation.apply(tt.tfms, sample)
end