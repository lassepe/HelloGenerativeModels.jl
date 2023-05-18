struct VAE
    z_dim::Integer
    distribution_encoder::Any
    sample_decoder::Any
end

@functor VAE (distribution_encoder, sample_decoder)

function (m::VAE)(x, ϵ)
    d = m.distribution_encoder(x)
    z = d.μ + d.Σ_diag .* ϵ
    x̂ = m.sample_decoder(z)
    (; x̂, z, d)
end

function sample!(rng::AbstractRNG, m::VAE, n_samples::Integer, device=cpu)
    ϵ = randn(rng, m.z_dim, n_samples) |> device
    device(m.sample_decoder)(ϵ)
end

function mvnormal_parameters(x)
    in_dim = size(x, 1)
    iseven(in_dim) || throw(ArgumentError("x must have an even first dimension for splitting."))
    out_dim = in_dim ÷ 2
    μ = selectdim(x, 1, 1:out_dim)
    Σ_diag = softplus.(selectdim(x, 1, (out_dim+1):in_dim))
    (; μ, Σ_diag)
end

function kld_from_normal(μ, Σ_diag)
    k = length(μ)
    1 // 2 * (sum(Σ_diag) + sum(μ .* μ) .- k .- sum(log.(prod(Σ_diag; dims=1))))
end

function get_default_loss(model::VAE; device, rng)
    function loss(x)
        ϵ = randn(rng, model.z_dim, size(x, 2)) |> device
        x̂, _, d = model(x, ϵ)
        D_kl = kld_from_normal(d.μ, d.Σ_diag)

        (sum((x .- x̂) .^ 2) + D_kl) / size(x, 2)
    end
end

function setup_vae(; setup=get_default_setup())
    distribution_encoder = Chain(
        Dense(setup.dims.data, setup.dims.hidden, sin),
        Dense(setup.dims.hidden, 2 * setup.dims.z),
        mvnormal_parameters,
    )
    sample_decoder = Chain(Dense(setup.dims.z, setup.dims.hidden, sin), Dense(setup.dims.hidden, setup.dims.data))
    VAE(setup.dims.z, distribution_encoder, sample_decoder) |> setup.training_config.device
end

function train_vae(; vae, setup)
    for epoch in 1:setup.training_config.n_epochs
        println("Epoch $epoch")
        loss = get_default_loss(vae; setup.training_config.device, setup.rng)
        @info "loss: $(loss(setup.dataset))"
        Flux.train!(loss, params(vae), setup.data_batch_iterator, setup.training_config.optimizer)
    end
end

function visualize_samples(samples, name)
    df = [(; x, name) for x in cpu(samples)[:]]
    data(df) * mapping(:x; color=:name) * density()
end

function vae_demo(; setup=get_default_setup())
    vae = setup_vae(; setup)
    train_vae(; vae, setup)
    vae_samples = sample!(setup.rng, vae, 1000)
    (visualize_samples(vae_samples, "VAE") + visualize_samples(setup.dataset, "GT")) |> draw |> display
    (; vae, setup)
end
