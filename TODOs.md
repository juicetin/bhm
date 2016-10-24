* get 'even' samples from multi-labels as well as the reduced versions
* see how predictions are when doing it this way compared to the usual 10-fold cross validation
* assess full query predictions when using this even-multi-label version
* get (almost) equally balanced 
* get a low average variance over even ('even' differs) splits - overall trend : lower variance, more even areas
    + lower variance -> 'better' results
* calculate r-hat statistic
* fix MCMC weight histograms to auto-split onto the appropriate number of pages
* fix all graphs (axes, colorbars, labels, titles, etc.) and figures

* [FIXED] redo GP predictions - hyperparameters should NOT all be 1
    + didn't optimize gpy model, assumption that it was executed on calling predict was incorrect
* do 'subsampling' of the multi-label data to even out distributions similar to how the 1000 indices for GP Regression was originally done - may improve performance

