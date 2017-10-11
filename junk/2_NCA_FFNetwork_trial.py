
import tensorflow as tf

#%%
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

#%%
# FFNN hyperparams
#==============================================================================

# These should be added to graph class attribs
DEPTH = 2
MAXWIDTH = 200
MINWIDTH = 50
NONLIN = "Tanh"

# Convert to integer/bool 
#(important for BayesOpt to work properly since it tries float values)  
DEPTH = int(DEPTH)
MAXWIDTH = int(MAXWIDTH)
MINWIDTH = int(MINWIDTH)

#%%
# Define sizes of weights and biases
#==============================================================================

dim_in = dim_input

if DEPTH == 1:
    dim_out = MINWIDTH
else:
    dim_out = MAXWIDTH

weights_sizes = {'layer_1': [dim_in, dim_out]}
biases_sizes = {'layer_1': [dim_out]}
dim_in = dim_out

if DEPTH > 2:
    for i in range(2, DEPTH):                
        dim_out = int(dim_out)
        weights_sizes['layer_{}'.format(i)] = [dim_in, dim_out]
        biases_sizes['layer_{}'.format(i)] = [dim_out]
        dim_in = dim_out
 
if DEPTH > 1:
    dim_out = MINWIDTH
    weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, dim_out]
    biases_sizes['layer_{}'.format(DEPTH)] = [dim_out]
    dim_in = dim_out
    
#%%
# Define layers
#==============================================================================

def _add_layer(layer_name, Input, APPLY_NONLIN = True,
               Mode = "Encoder", Drop = True):
    
    """ adds a single fully-connected layer"""
    
    with tf.variable_scope(layer_name):
        
        # initialize using xavier method
        
        m_w = weights_sizes[layer_name][0]
        n_w = weights_sizes[layer_name][1]
        m_b = biases_sizes[layer_name][0]
        
        xavier = tf.contrib.layers.xavier_initializer()
        
        w = tf.get_variable("weights", shape=[m_w, n_w], initializer= xavier)
        #variable_summaries(w)
     
        b = tf.get_variable("biases", shape=[m_b], initializer= xavier)
        #variable_summaries(b)
            
        # Do the matmul and apply nonlin
        
        with tf.name_scope("pre_activations"):   
            if Mode == "Encoder":
                l = tf.add(tf.matmul(Input, w),b) 
            elif Mode == "Decoder":
                l = tf.matmul(tf.add(Input,b), w) 
            #tf.summary.histogram('pre_activations', l)
        
        if APPLY_NONLIN:
            if NONLIN == "Sigmoid":  
                l = tf.nn.sigmoid(l, name= 'activation')
            elif NONLIN == "ReLU":  
                l = tf.nn.relu(l, name= 'activation')
            elif NONLIN == "Tanh":  
                l = tf.nn.tanh(l, name= 'activation') 
            #tf.summary.histogram('activations', l)
        
        # Dropout
        
        if Drop:
            with tf.name_scope('dropout'):
                l = tf.nn.dropout(l, keep_prob=1-DROPOUT_FRACTION)
            
        return l

#%%
# Now add the layers
#==============================================================================
    
with tf.variable_scope("FFNetwork"):
    
    l_in = X_input
 
    layer_params = {'APPLY_NONLIN' : True,
                    'Mode' : "Encoder",
                    'Drop' : True,
                    }
               
    for i in range(1, DEPTH):
         l_in = _add_layer("layer_{}".format(i), l_in, **layer_params)
         
    # outer layer (final, transformed datset)
    layer_params['Drop'] = False
    X_transformed = _add_layer("layer_{}".format(DEPTH), l_in, **layer_params)