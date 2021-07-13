include("../imports.jl")


@testset ExtendedTestSet "`ProjectiveTransforms`" begin
    @testset ExtendedTestSet "image" begin
        encoding = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)
        block = FastAI.Image{2}()

        ## We run `ProjectiveTransforms` in the different [`Context`]s:
        imagetrain = encode(encoding, Training(), block, image)
        @test size(imagetrain) == (32, 32)

        imagevalid = encode(encoding, Validation(), block, image)
        @test size(imagevalid) == (32, 32)

        imageinference = encode(encoding, Inference(), block, image)
        @test size(imageinference) == (32, 48)

        ## During inference, the aspect ratio should stay the same
        @test size(image, 1) / size(image, 2) == size(imageinference, 1) / size(imageinference, 2)
    end

    @testset ExtendedTestSet "keypoints" begin
        encoding = ProjectiveTransforms((32, 48))
        ks = [SVector(0., 0), SVector(64, 96)]
        block = FastAI.Keypoints{2}(10)
        bounds = DataAugmentation.Bounds((1:32, 1:48))
        r = DataAugmentation.getrandstate(encoding.tfms[Training()])
        kstrain = encode(encoding, Training(), block, ks; state = (bounds, r))
        ksvalid = encode(encoding, Validation(), block, ks; state = (bounds, r))
        ksinference = encode(encoding, Inference(), block, ks; state = (bounds, r))

        @test kstrain == ksvalid == ksinference
    end

    @testset ExtendedTestSet "image and keypoints" begin
        encoding = ProjectiveTransforms((32, 32))
        image = rand(RGB, 64, 96)
        ks = [SVector(0., 0), SVector(64, 96)]
        blocks = (FastAI.Image{2}(), FastAI.Keypoints{2}(10))

        @test_nowarn encode(encoding, Training(), blocks, (image, ks))
        @test_nowarn encode(encoding, Validation(), blocks, (image, ks))
        @test_nowarn encode(encoding, Inference(), blocks, (image, ks))
    end

    #= depends on buffered interface
    @testset ExtendedTestSet "`run!`" begin
        encoding = ProjectiveTransforms((32, 32))
        image1 = rand(RGB, 64, 96)
        image2 = rand(RGB, 64, 96)
        buf = encode(encoding, Validation(), image1)
        cbuf = copy(buf)
        encode!(buf, encoding, Validation(), image1)
        # run and run! should return the same result
        @test buf ≈ cbuf

        encode!(buf, encoding, Validation(), image2)
        # buf should be modified on different input
        @test !(buf ≈ cbuf)
    end
    =#
end
