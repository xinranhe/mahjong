import tensorflow as tf

from single_efficiency.transformer import Transformer

class Model(object):
    def __init__(self, params):
        self.params = params
        
        self.gamma = self.params["gamma"]
        
        # Input place holder (s_t, s_{t+1})
        # S_t
        self.state = tf.placeholder(dtype=tf.int32, shape=(None, 14), name="state")
        # S_{t+1}
        self.state_plus = tf.placeholder(dtype=tf.int32, shape=(None, 14), name="state_plus")
        # r
        self.reward = tf.placeholder(dtype=tf.float32, shape=(None), name="reward")
        # is S_{t+1} terminal state
        self.is_terminal = tf.placeholder(dtype=tf.float32, shape=(None), name="is_terminal")
        # a 
        self.action = tf.placeholder(dtype=tf.int32, shape=(None), name="action")
        # target
        self.target = tf.placeholder(dtype=tf.float32, shape=(None), name="target")
        
        self.target_net = Transformer(False, self.params, "target_net")
        self.eval_net = Transformer(True, self.params, "eval_net")
        # replacement op to copy params from eval to target network
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        self.prediction = self.eval_net(self.state)
        current_Q = tf.gather(self.prediction, self.action, axis=1)
        next_max_Q = tf.reduce_max(self.target_net(self.state_plus), axis=1)
        
        self.target_tensor = self.reward + self.gamma * next_max_Q * (1.0 - self.is_terminal)
        self.loss = tf.nn.l2_loss(tf.clip_by_value(self.target - current_Q, -5, 5))
        
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            self.params["learning_rate"],
            beta1=self.params["optimizer_adam_beta1"],
            beta2=self.params["optimizer_adam_beta2"],
            epsilon=self.params["optimizer_adam_epsilon"])

        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(self.loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        self.train_op = tf.group(minimize_op, update_ops)
    
    def init(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
    
    def input_to_feed_dict(self, input_tuple):
        return {
            self.state: input_tuple[0],
            self.state_plus: input_tuple[1],
            self.reward: input_tuple[2],
            self.action: input_tuple[3],
            self.is_terminal: input_tuple[4]
        }

    def train(self, sess, input_tuple):
        feed_dict = self.input_to_feed_dict(input_tuple)
        target = sess.run(self.target_tensor, feed_dict=feed_dict)
        feed_dict[self.target] = target
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss
    
    def inference(self, sess, input_state):
        return sess.run(self.prediction, feed_dict={self.state: [input_state]})
    
    def update_target_network(self, sess):
        sess.run(self.target_replace_op)
