include("../imports.jl")


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
        transform = ProjectiveTransforms((32, 48))
        ks = [SVector(0., 0), SVector(64, 96)]
        keypoints = Keypoints(ks, (64, 96))
        kstrain = FastAI.run(transform, Training(), keypoints)
        ksvalid = FastAI.run(transform, Validation(), keypoints)
        ksinference = FastAI.run(transform, Inference(), keypoints)

        @test kstrain == ksvalid == ksinference
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
        # buf should be modified on different input
        @test !(buf ≈ cbuf)
    end
end
