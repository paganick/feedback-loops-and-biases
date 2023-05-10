from scipy.stats import truncnorm
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import numpy as np
from scipy.stats import bernoulli


def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def x2y(x, bias, std):
    x_error = get_truncated_normal(np.array(x)+np.array(bias), std, 0, 1).rvs()
    y = bernoulli.rvs(x_error, size=1)[0]
    return y


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    #setp(bp['fliers'][0], color='blue')
    #setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

