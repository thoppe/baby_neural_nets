import tensorflow as tf
import numpy as np
import itertools
import os
import h5py
import sys

name = sys.argv[1]

save_dest = 'images/{}'.format(name)
os.system('rm -fr images/{}'.format(name))

os.system('mkdir -p images')
os.system('mkdir -p images/{}'.format(name))

f_h5 = 'image_data.h5'
h5 = h5py.File(f_h5,'r')

if name not in h5:
    msg = "Model {} not known in {}".format(name,f_h5)
    raise ValueError(msg)


group = h5[name]
image_data   = (group['labels'][:])
image_colors = group["label_colors"][:]

# The nmumber of classes is number of label colors
n_classes = len(image_colors)

# Dim is the embedding dimension (2D in this case)
dim = 2

_MLP_name_counts = itertools.count(0)

total_epochs    = 100000
draw_interval   = 2
report_interval = 5
draw_n = 3000
learning_rate = 1e-3
dropout_rate = 0.75

N   = 5000
n_layers = 10
m_size   = 40
use_dropout = False

layer_shapes = [m_size]*n_layers + [n_classes,]
dropout_keep_prob = tf.placeholder(tf.float32)

activation_func = tf.nn.tanh
#activation_func = tf.nn.relu
#activation_func = tf.nn.sigmoid
#activation_func = None

init_func = tf.uniform_unit_scaling_initializer
#init_func = tf.contrib.layers.xavier_initializer
#init_func = tf.truncated_normal_initializer
#init_kwargs = {'stddev':0.1}
init_kwargs = {'factor':1.44}


def MLP_layer(x, output_shape, name=None, dropout=False):
    '''
    Does not include an activation function.
    '''
    if name is None:
        name = "MLP_{}".format(_MLP_name_counts.next())

    input_shape = x.get_shape().as_list()[1]
    
    with tf.variable_scope(name):

        Wsp = [input_shape, output_shape]
        W = tf.get_variable("W", shape=Wsp,
                            initializer=init_func(**init_kwargs))
        b = tf.get_variable("b", shape=[output_shape])

        if dropout:
            W = tf.nn.dropout(W, dropout_keep_prob)

        mlp = tf.add(tf.matmul(x, W), b)
        
    return mlp

def simple_MLP(input_dim, layer_shapes,
               activation_function=None):

    x = tf.placeholder("float", [None, input_dim])
    layers = [x,]
    dropout = use_dropout
    
    for k,n in enumerate(layer_shapes):

        # Apply dropout to last layer
        #if k == len(layer_shapes)-1:
        #    dropout = True

        print "Building a layer of size {}, dropout {}".format(n, dropout)

        mlp = MLP_layer(layers[-1], n, dropout=dropout)

        if activation_function is None:
            layers.append(mlp)
        else:
            layers.append(activation_function(mlp))

    return layers


layers = simple_MLP(dim, layer_shapes, activation_function=activation_func)

# Throw on an activation function on the last layer
layers[-1] = activation_func(layers[-1])

# Construct model
model = layers[-1]
X = layers[0]

# Define loss and optimizer
labels = tf.placeholder(dtype=tf.int32, shape=[None,], name="output")
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(model,labels,name='cost')

cost_mean = tf.reduce_mean(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost_mean)
optimizer =  tf.train.AdamOptimizer(learning_rate).minimize(cost_mean)

init = tf.initialize_all_variables()

dx = 1.0
xi, yi = np.linspace(-dx,dx, draw_n), np.linspace(-dx,dx, draw_n)
xi, yi = np.meshgrid(xi, yi)
test_pts = np.array(zip(xi.ravel(), yi.ravel()))

def generate_data(zi=None):
    dimx,dimy = image_data.shape

    xidx = np.random.randint(0, high=dimx, size=(N,))
    yidx = np.random.randint(0, high=dimy, size=(N,))
    labels = image_data[xidx,yidx].astype(np.int32)
    pt_x = xidx.astype(np.float32)*(2.0/dimx) - 1
    pt_y = yidx.astype(np.float32)*(2.0/dimy) - 1
    pts = np.vstack([pt_y, pt_x]).T

    return pts,labels


# Drawing point
import pylab as plt
fig = plt.figure(frameon = False)
fig.set_size_inches(4, 4)
ax = plt.Axes(fig, [0., 0., 1., 1.], )
ax.set_axis_off()
fig.add_axes(ax)


img_count = 0
def draw_boundry(n):
    print "Drawing",n
    args = {X:test_pts, dropout_keep_prob:1.0}
    mx, = sess.run([model], feed_dict=args)
    
    img_idx = np.argmax(mx,axis=1)
    zi = image_colors[img_idx].reshape((xi.shape[0],xi.shape[1],4))

    '''
    prob = np.exp(mx)
    prob /= prob.sum(axis=1).reshape(-1,1)

    channels = np.array([prob[:,i]*image_colors[i].reshape(-1,1)
                         for i in range(n_classes)])
    zi = (channels.mean(axis=0).T * 255).astype(np.int32)
    zi = zi.reshape((xi.shape[0],xi.shape[1],4))
    '''
    
    plt.cla()
    plt.clf()
    plt.imshow(zi,
               #vmin=0,
               #vmax=255,
               interpolation='none',
               origin='upper',
               aspect='auto',
               extent=[-dx,dx,-dx,dx],
               #cmap="Reds"
    )
    plt.xlim(-dx,dx)
    plt.ylim(-dx,dx)
    plt.axis('off')

    global img_count
    n = img_count
    img_count += 1

    #plt.savefig('images/{:d}.png'.format(n),bbox_inches='tight')
    f_png = os.path.join(save_dest, '{:d}.png'.format(img_count))
    plt.savefig(f_png,dpi=100,pad_inches=0)

    return zi


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    zi = None

    for epoch_n in range(total_epochs):

        X_pts, Y_pts = generate_data(zi)
       
        args = {X:X_pts, labels:Y_pts, dropout_keep_prob:dropout_rate}
        #opt, c, mx = sess.run([optimizer, cost_mean, model], feed_dict=args)
        sess.run([optimizer,], feed_dict=args)

        if epoch_n % draw_interval==0:
            zi = draw_boundry(epoch_n)

        if epoch_n % report_interval==0:
            c, mx = sess.run([cost_mean, model], feed_dict=args)
            guess = np.argmax(mx,axis=1)
            acc = (guess == Y_pts).sum() / float(len(Y_pts))
            print "{:08d} {:0.6f} {:0.6f}".format(epoch_n, c, acc)

    draw_boundry(epoch_n)
    #plt.show()

