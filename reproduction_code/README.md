# Reproduction code


## Setup

Start the Julia REPL (Julia version 1.5) and type `]` to activate the package manager. Then type and enter:
```{julia}
instantiate
```
This will automatically load all required julia dependencies. 

For the `cll.jl` file, a R installation (it will be called through `RCall`) is also required with an installation of the `MOFAdata` package.
This package may be installed as follows:

```{r}
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("MOFAdata")
```

## File description

* motivation.jl
* oracle_risks.jl: 
* cll.jl