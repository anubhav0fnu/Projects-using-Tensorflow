
# coding: utf-8

# ### Minimize the non-convex Rosenbrock function

# In[33]:


import tensorflow as tf
import argparse
def createGraph(a,b,lr,optimiser ):
    tf.reset_default_graph()
    w1 = tf.get_variable(dtype=tf.float32, shape=(), name="w1",                        initializer=tf.random_normal_initializer(0.0,1.0))
    w2 = tf.get_variable(dtype=tf.float32, shape=(), name="w2",                        initializer=tf.random_normal_initializer(0.0,1.0))
    a = tf.constant(a , name="a")
    b = tf.constant(b,  name="b")
    costFunc = tf.add( tf.square(                                 tf.subtract( a,                                             w1,                                             name='term1'),
                                name='term1_sq'),\
                   tf.multiply( b,\
                                tf.square(   tf.subtract(  \
                                                            w2,\
                                                            tf.square(  w1,\
                                                                        name='weight1_sq'),\
                                                            name='term2'),\
                                             name='term2_sq'),\
                                name='term2_Sq_Mul_b'),\
                   name='costFunc') 
    print(a,b)
    print(lr)
    print(optimiser)
    if 'gd'==optimiser:
        train_step =tf.train.GradientDescentOptimizer(learning_rate=lr, name='GradientDescent').minimize(costFunc, name= 'train_step')
    if 'gdm'==optimiser:
        train_step =tf.MomentumOptimizer(learning_rate=lr,momentum=0.9, name='Momentum').minimize(costFunc, name= 'train_step')
    if 'adam'==optimiser:
        train_step =tf.AdamOptimizer(learning_rate=lr,name='Adam').minimize(costFunc, name= 'train_step')
    
    init = tf.global_variables_initializer()
    #once you build the graph, write to file
    file_writer= tf.summary.FileWriter("./datasets/myTensorboardLogs/HW1_Oct11/", tf.get_default_graph())
    
    sess = tf.Session()
    sess.run(fetches=[init])
    for x in range(10):
        acc= sess.run(fetches=[train_step,w1,w2])
        print('Accuracy at step %s: %s' % (x, acc))


# In[34]:


def main():
    parser = argparse.ArgumentParser(description='It Minimize the non-convex Rosenbrock function using constants a & b, a learning rate, and an optimiser ')
# now you need to fillin the argParser
    parser.add_argument('a', metavar='constant', type=float, nargs='+',                   help='an floating point number--constant')
    parser.add_argument('b', metavar='constant', type=float, nargs='+',                   help='an floating point number--constant')
    parser.add_argument('lr', metavar='variable', type=float, nargs='+',                   help='an floating point number--learning rate')
    parser.add_argument('optimizer',choices=['gd', 'gdm', 'adam'], metavar='variable', type=str, nargs='+',                   help='the optimization algorithm')
    args= parser.parse_args()
#     return a dict
#     weight1=0
#     weight1=0
#     weight1=0
    createGraph(args.a, args.b, args.lr, args.optimizer)

if __name__ == '__main__':
    main()

