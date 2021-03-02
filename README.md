# AdaSmooth
An optimizer that smoothly transitions from SGD to Adam via weighted average of calculated steps

A lot of the talk about making an optimizer as fast as Adam that generalizes as well as SGD revolves around the idea that Adam does not generalize as well in later stages of optimization. We now know that this is to due to the adaptive parameters initializing to poor values in the early stages, specifically because the moving averages are so volatile and heavily influenced by the first few steps. Given that we use a running average for computational efficiency instead of a true windowed moving average, outliers can corrupt these moving averages and make it impossible for the optimizer to recover. It's easy to see why this is the case, as any outliers in gradient trajectory can essentially break the moving average for many iterations to come. The bigger the outlier, the more steps it will take for it to be averaged out. It's a vicious cycle where poor estimates of per-parameter gradient momenta causes a suboptimal step which causes more bad gradient information and so on. The end result is a poor optimization trajectory that negatively affects final performance.

One of the ways we currently remedy this is to add a warmup phase where we use a small step size in the early stages to allow adaptive params to initialize to steady values before ramping up the step size and getting the convergence speed benefits of Adam.

However, there had been studies that show Adam actually outperforms SGD in final generalization as well (specifically for larger networks), given that adaptive params are initialized to good values. Recently I've been experimenting with the idea of transitioning from SGD to an Adaptive optimizer by doing a weighted average of a calculated SGD and Adam step, where the SGD step is weighted less as optimization goes on. This allows for Adam to initialize to good values as it is guided by SGD. While there is still the issue of moving averages being corruptable by outliers, this will not manifest itself in the optimization trajectory until later in optimization, when these outliers should be averaged out anyway by going through many good optimization steps provided by the SGD step weighting.

