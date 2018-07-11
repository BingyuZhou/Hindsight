from DDPG import DDPG
import tensorflow as tf
import numpy as np


def train(actor, critic, env, params, num_epoch, num_cycle, num_episode,
          num_rollout, num_train, replaybuffer, state_shape, action_shape,
          action_range):

    discount = params['discount']
    decay = params['decay']
    batch_size = params['batch_size']
    lr_actor = params['lr_actor']
    lr_critic = params['lr_critic']
    eps = params['eps']

    agent = DDPG(actor, critic, replaybuffer, params, state_shape,
                 action_shape, action_range)
    global_step = tf.train.get_or_create_global_step()
    with tf.Session() as sess:
        agent._initialize(sess)
        writer = tf.summary.FileWriter('./summary/', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for n_e in range(num_epoch):
            for n_cy in range(num_cycle):
                for n_ep in range(num_episode):
                    # Sample training data set
                    for n_roll in range(num_rollout):

                        s = np.copy(env.state)
                        a, _ = agent.pi(s, eps_greedy=True, compute_V=True)

                        s_next = env.update_state(a)
                        r = env.reward(s_next)

                        replaybuffer.add(s, env.goal, a, s_next, r)

                        if (r == 0):
                            env.reset()
                            break
                for n_i in range(num_train):
                    critic_ls, actor_ls = agent.train(global_step)

                    if (n_i % 2 == 0):
                        tf.logging.info(
                            'Epoch {0} Cycle {1} Iteration {2}: critic_ls {3:.2g}, actor_ls {4:.2g}'.
                            format(n_e, n_cy, n_i, critic_ls, actor_ls))

                agent.update_target_nn()

        writer.close()
        saver = tf.train.Saver()
        saver.save(sess, '/tmp/model.ckpt')
