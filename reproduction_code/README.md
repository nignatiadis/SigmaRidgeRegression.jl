# Reproduction code

## Setup

Requirements:
* `Julia` language  (code has been tested on Julia version 1.6-rc1, but Julia 1.5 should also work) 
* `R` language with the [seagull](https://cran.r-project.org/web/packages/seagull/index.html) package. It can be installed from within `R`, for example using `install.packages("seagull")`. The `R` installation should be detectable from `Julia` using the [RCall.jl](https://juliainterop.github.io/RCall.jl/dev/) package.
* Once the above steps have been completed, the rest of the required Julia packages may be installed as follows: Start the `Julia` REPL in this folder and type `]` to activate the package manager. Then type and enter

```julia
activate .
```
followed by
```julia
instantiate
```
This will automatically install all required Julia dependencies. 



## File description

* `motivation.jl`: Code to reproduce Figures 1 and 2.
* `oracle_risks.jl`: Code to reproduces Figures 4 and S1.
* `cll.jl`: Code to reproduce Tables 1 and 2.
* `million_songs.jl`: Code to reproduce Figures 5 and 6.
* `simulations.jl` and `simulations_plots.jl`: Code to reproduce Figure 7 of the manuscript. Note that you call `simulations.jl` for example via `julia simulations.jl 1` (and similarly up to `6`) and this generates files in `simulation_results`. These results are then loaded by `simulations_plots.jl`.

## Datasets 

There are two datasets that are used in this paper.

* **CLL**: The first dataset (CLL) is already available from within the `SigmaRidgeRegression.jl` package and can be loaded via `CLLData.load()`, also see the 
`dataset/cll_data_preprocessing.jl` file in this repo for details on how this dataset was imported from `R`.
* **Million Songs Dataset**: This file is 200 MB, so it needs to be manually downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). In particular, navigate 
to `Data Folder` at that link, download `YearPredictionMSD.txt.zip` and extract the contents of the zip in the `dataset` folder herein.


