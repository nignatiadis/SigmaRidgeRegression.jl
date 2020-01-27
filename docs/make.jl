using Documenter, RidgeRegression

makedocs(;
    modules=[RidgeRegression],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/nignatiadis/RidgeRegression.jl/blob/{commit}{path}#L{line}",
    sitename="RidgeRegression.jl",
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/nignatiadis/RidgeRegression.jl",
)
