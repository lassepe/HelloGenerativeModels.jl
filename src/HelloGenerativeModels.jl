module HelloGenerativeModels

using CUDA: CUDA
using Flux:
    @functor,
    Chain,
    Dense,
    Flux,
    cpu,
    gpu,
    logitbinarycrossentropy,
    mse,
    params,
    relu,
    softplus
using AlgebraOfGraphics: AlgebraOfGraphics, data, draw, mapping, visual, density
using Random: Random, AbstractRNG
using Zygote: Zygote
using GLMakie: GLMakie

function get_default_setup()
    function decoder_gt(z)
        tanh.(1.5z)
    end
    rng = Random.MersenneTwister(1)
    training_config = (;
        optimizer=Flux.ADAM(0.001),
        n_epochs=100,
        batchsize=100,
        n_datapoints=100_000,
        device=cpu
    )

    dims = (; data=1, hidden=100, z=1)
    dataset = randn(rng, dims.z, training_config.n_datapoints) |> decoder_gt |> training_config.device
    data_batch_iterator = Flux.Data.DataLoader(dataset; training_config.batchsize)

    (; rng, training_config, dims, dataset, data_batch_iterator)
end


include("vae.jl")

end # module HelloGenerativeModels
