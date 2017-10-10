
import tensorflow as tf

#%% ===========================================================================
# Placeholders
# =============================================================================

dim_input = 399
tf.reset_default_graph()
X_input = tf.placeholder("float", [None, dim_input], name='X_input')
Pij_mask = tf.placeholder("float", [None, None], name='Pij_mask')

# Hyperparams
ALPHA = tf.placeholder(tf.float32, name='ALPHA')
LAMBDA = tf.placeholder(tf.float32, name='LAMDBDA')
SIGMA = tf.placeholder(tf.float32, name='SIGMA')
DROPOUT_FRACTION = tf.placeholder(tf.float32, name='DROPOUT_FRACTION')

#%%============================================================================
# FFNN hyperparams
#==============================================================================

DEPTH = int(DEPTH)
MAXWIDTH = int(MAXWIDTH)
dim_out