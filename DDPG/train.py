from DDPG import DDPG
import tensorflow as tf
import numpy as np


def train(actor, critic, env, env_evl, params, num_epoch, num_cycle,
          num_episode, num_rollout, num_train, replaybuffer, state_shape,
          action_shape, action_range):

    discount = params['discount']
    decay = params['decay']
    batch_size = params['batch_size']
    lr_actor = params['lr_actor']
    lr_critic = params['lr_critic']
    eps = params['eps']

    agent = DDPG(actor, critic, replaybuffer, params, state_shape,
                 action_shape, action_range)

    with tf.Session() as sess:
        agent._initialize(sess)

        writer = tf.summary.FileWriter('./summary/', sess.graph)

        tf.logging.set_verbosity(tf.logging.INFO)

        for n_e in range(num_epoch):
            for n_cy in range(num_cycle):
                for n_ep in range(num_episode):
                    # Sample training data set
                    for n_roll in range(num_rollout):

                        s = np.copy(env.state)
                        state = np.concatenate(
                            (s.reshape(1, -1), env.goal.reshape(1, -1)),
                            axis=1)
                        a, _ = agent.pi(state, eps_greedy=True, compute_V=True)

                        s_next = env.update_state(action_range[1] * a)
                        r = env.reward(s_next)

                        replaybuffer.add(s, env.goal, a, s_next, r)

                        if (r == 0):
                            env.reset()
                            break
                for n_i in range(num_train):
                    critic_ls, actor_ls = agent.train()

                    if (n_i % 2 == 0):
                        tf.logging.info(
                            'Epoch {} Cycle {} Iteration {}: critic_ls {:.2g}, actor_ls {:.2g}'.
                            format(n_e, n_cy, n_i, critic_ls, actor_ls))

                agent.update_target_nn()

                # Evaluate
                tf.logging.info('-------Evaluate---------')
                tf.logging.info('Q value: ')
                for n_roll in range(num_rollout):
                    s = np.copy(env.state)
                    state = np.concatenate(
                        (s.reshape(1, -1), env.goal.reshape(1, -1)), axis=1)
                    a, Q = agent.pi(state, False, True)
                    tf.logging.info('{} '.format(Q))
                    env_evl.update_state(action_range[1] * a)
                    reward = env_evl.reward(env_evl.state)
                    if reward == 0:
                        break

                if reward == 0:
                    tf.logging.info('Success')
                else:
                    tf.logging.info('Fail')
                env_evl.reset()

        writer.close()
        saver = tf.train.Saver()
        saver.save(sess, '/tmp/model.ckpt')
