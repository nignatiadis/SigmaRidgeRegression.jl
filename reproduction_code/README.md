# Reproduction code


## Setup

Start the `Julia` REPL (Julia version 1.5) in this folder and type `]` to activate the package manager. Then type and enter

```julia
activate .
```
followed by
```julia
instantiate
```
This will automatically install all required Julia dependencies. 

For the `cll.jl` file, a R installation (it will be called through `RCall`) is also required with an installation of the `MOFAdata` package.
This package may be installed from within `R` as follows:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("MOFAdata")
```

## File description

* `motivation.jl`: Code to reproduce Figures 1 and 2.
* `oracle_risks.jl`: Code to reproduces Figures 4 and S1.
* `cll.jl`: Code to reproduce Tables 1 and 2.
* `million_songs.jl`: Code to reproduce Figures 5 and 6.
* `simulations.jl` and `simulations_plots.jl`: Code to reproduce Figure 7 of the manuscript. Note that you call `simulations.jl` for example via `julia simulations.jl 1` (and similarly up to `6`) and this generates files in `simulation_results`. These results are then loaded by `simulations_plots.jl`.