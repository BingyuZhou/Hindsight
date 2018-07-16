import argparse
import numpy as np
import tensorflow as tf
from train import train
from DDPG import DDPG
from model import Critic, Actor
from bitflipping import bitflipping as bf
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt


def plot_ls(critic_ls_all, actor_ls_all):
    plt.figure()
    plt.plot(critic_ls_all, label='critic_ls')
    plt.plot(actor_ls_all, label='actor_ls')
    plt.legend()
    plt.show()


def main(num_bit, hid_layer_critic, hid_layer_actor, num_epoch, num_cycle,
         num_episode, num_train):
    """
    High level configuration of learning process
    """

    # Play environment
    env = bf(num_bit)
    env_evl = bf(num_bit)

    # Buffer
    replaybuffer = ReplayBuffer(5000, num_bit)

    # Architecture for models
    critic = Critic(hid_layer_critic, 'critic', False)
    actor = Actor(hid_layer_actor, 1, 'actor', False)

    # Training
    params = {}
    params['discount'] = 0.98
    params['decay'] = 0.8
    params['batch_size'] = 32
    params['lr_actor'] = 0.001
    params['lr_critic'] = 0.001
    params['eps'] = 0.2

    num_rollout = num_bit
    state_shape = 2 * num_bit
    action_shape = 1
    action_range = [0, num_bit - 1]

    tf.logging.info('*******Start training*********')
    critic_ls_all, actor_ls_all = train(
        actor, critic, env, env_evl, params, num_epoch, num_cycle, num_episode,
        num_rollout, num_train, replaybuffer, state_shape, action_shape,
        action_range)

    plot_ls(critic_ls_all, actor_ls_all)

    # Testing
    # tf.logging.info('********Testing********')
    # env_test = bf(num_bit)
    # agent = DDPG(actor, critic, replaybuffer, params, state_shape,
    #              action_shape, action_range)

    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     tf.logging.infor('restore model...')
    #     saver.restore(sess, '/tmp/model.ckpt')

    #     success = 0
    #     for i in range(100):
    #         for j in range(num_bit):
    #             a, Q = agent.pi(env_test.state, False, True)

    #             env_test.update_state(a)
    #             reward = env_test.reward(env_test.state)

    #             if reward == 0:
    #                 success += 1
    #                 break
    #         env_test.reset()
    #     tf.logging.info('success rate {}'.format(success / 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--num-bit', type=int, help='number of bits', default=5, required=True)
    parser.add_argument(
        '--hid-layer-critic', nargs='+', default=[1], required=True)
    parser.add_argument(
        '--hid-layer-actor', nargs='+', default=[1], required=True)
    parser.add_argument('--num-epoch', type=int, default=10, required=True)
    parser.add_argument('--num-cycle', type=int, default=5, required=True)
    parser.add_argument('--num-episode', type=int, default=16, required=True)
    parser.add_argument('--num-train', type=int, default=30, required=True)

    args = parser.parse_args()
    dict_args = vars(args)
    main(**dict_args)
