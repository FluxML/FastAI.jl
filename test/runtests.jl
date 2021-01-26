using FastAI
using DLPipelines
using DataAugmentation
using DataAugmentation: getbounds
using Colors: RGB
using Test
using TestSetExtensions
using StaticArrays

##

@testset ExtendedTestSet "`ProjectiveTransforms`" begin
    @testset ExtendedTestSet "image" begin
        transform = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)

        ## We run `ProjectiveTransforms` in the different [`Context`]s:
        imagetrain = FastAI.run(transform, Training(), image)
        @test size(imagetrain) == (32, 32)

        imagevalid = FastAI.run(transform, Validation(), image)
        @test size(imagevalid) == (32, 32)

        imageinference = FastAI.run(transform, Inference(), image)
        @test size(imageinference) == (32, 48)

        ## During inference, the aspect ratio should stay the same
        @test size(image, 1) / size(image, 2) == size(imageinference, 1) / size(imageinference, 2)
    end

    @testset ExtendedTestSet "keypoints" begin
        transform = ProjectiveTransforms((32, 32))
        ks = [SVector(0., 0), SVector(64, 96)]
        keypoints = Keypoints(ks, (64, 96))
        kstrain = FastAI.run(transform, Training(), keypoints)
        ksvalid = FastAI.run(transform, Validation(), keypoints)
        ksinference = FastAI.run(transform, Inference(), keypoints)

        @test ksvalid[1][1] == 0
        @test ksvalid[2][1] == 32
        @test ksinference[2] == ks[2] ./ 2
    end

    @testset ExtendedTestSet "image and keypoints" begin
        transform = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)
        ks = [SVector(0., 0), SVector(64, 96)]

        @test_nowarn FastAI.run(transform, Training(), (image, ks))
        @test_nowarn FastAI.run(transform, Validation(), (image, ks))
        @test_nowarn FastAI.run(transform, Inference(), (image, ks))
    end

    @testset ExtendedTestSet "`run!`" begin
        transform = ProjectiveTransforms((32, 32))
        image1 = rand(RGB, 64, 96)
        image2 = rand(RGB, 64, 96)
        buf = FastAI.run(transform, Validation(), image1)
        cbuf = copy(buf)
        FastAI.run!(buf, transform, Validation(), image1)
        # run and run! should return the same result
        @test buf ≈ cbuf

        FastAI.run!(buf, transform, Validation(), image2)
        # run and run! should return the same result
        @test !(buf ≈ cbuf)

    end
end

@testset ExtendedTestSet "ImagePreprocessing" begin
    step = ImagePreprocessing((0, 0, 0), (.5, .5, .5))
    image = rand(RGB, 100, 100)
    x = FastAI.run(step, Training(), image)
    @test size(x) == (100, 100, 3)
    @test eltype(x) == Float32

    image2 = rand(RGB, 100, 100)
    buf = copy(x)
    @test_nowarn FastAI.run!(buf, step, Training(), image2)
    @test !(buf ≈ x)
end


@testset ExtendedTestSet "`ImageClassification`" begin
    @testset ExtendedTestSet "Core interface" begin
        @testset ExtendedTestSet "`encodeinput`" begin
            method = ImageClassification(10, (32, 32))
            image = rand(RGB, 64, 96)

            xtrain = encodeinput(method, Training(), image)
            @test size(xtrain) == (32, 32, 3)
            @test eltype(xtrain) == Float32

            xinference = encodeinput(method, Inference(), image)
            @test size(xinference) == (32, 48, 3)
            @test eltype(xinference) == Float32
        end

        @testset ExtendedTestSet "`encodetarget`" begin
            method = ImageClassification(2, (32, 32))
            category = 1
            y = encodetarget(method, Training(), category)
            @test y ≈ [1, 0]
            encodetarget!(y, method, Training(), 2)
            @test y ≈ [0, 1]
        end

        @testset ExtendedTestSet "`encode`" begin
            method = ImageClassification(10, (32, 32))
            image = rand(RGB, 64, 96)
            category = 1
            @test_nowarn encode(method, Training(), (image, category))
        end
    end
end
