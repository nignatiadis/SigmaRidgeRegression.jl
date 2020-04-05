using Documenter, SigmaRidgeRegression

makedocs(;
    modules=[SigmaRidgeRegression],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/nignatiadis/SigmaRidgeRegression.jl/blob/{commit}{path}#L{line}",
    sitename="SigmaRidgeRegression.jl",
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/nignatiadis/SigmaRidgeRegression.jl",
)
