# example script to run and evaluate setfit based few shot learning on sentiment data
# unlike zero or few shot prompting setfit first learns a sentence transformer base.
# This base essentially learns to  differentiate sentences that belong to the same class
# from those that belong to different classes.

from setfit import Set