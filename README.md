# Ranger-tensorflow
Radam+lookahead implemented by tensorflow

# Usage
from ranger import Ranger

num_sample=1000

lr = tf.train.exponential_decay(1e-8, global_step, num_sample*10, 0.9, staircase=True)

optimize =  Ranger(learning_rate=lr,beta1=0.90,epsilon=1e-8)
