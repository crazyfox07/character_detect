# @Time    : 2019/1/17 14:09
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : tmp.py

import tensorflow as tf
a = tf.get_variable("ii", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
n = tf.constant(10)

def cond(a, n):
    return  a< n
def body(a, n):
    a = a + 1
    return a, n

a, n = tf.while_loop(cond, body, [a, n])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run([a, n])
    print(res)

pass