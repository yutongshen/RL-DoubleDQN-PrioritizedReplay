import gym
from src.rl import RL
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

def play(game, rl, iteration, human_transition=None, human_ctrl=False):
    human_ctrl_cnt   = 0
    human_a          = 0
    step = 0
    if not human_transition:
        human_transition = []
    else:
        human_ctrl   = False
        for t in human_transition:
            rl.store_transition(t[0], t[1], t[2], t[3])
            step += 1
        
    total_score      = []
    episode = 0
    start_train = False
    while episode < iteration:
        s = game.reset()
        done = False
        descent = 1
        performance = 0
        while not done:
            game.render()
            a = rl.actor(s)
            if human_ctrl:
                if not human_ctrl_cnt:
                    try:
                        human_a = int(input('input action: '))
                        if human_a < 0 or human_a > 2:
                            human_a = 1
                    except ValueError:
                        human_a = 1
                    human_ctrl_cnt = 20
                else:
                    human_ctrl_cnt -= 1
                a = human_a

            s_, r, done, _ = game.step(a)
            performance += r
            position, velocity = s_
            r = abs(position + .52) / 1.12 + abs(velocity) / .07 - 1
            if done: r = 10
            rl.store_transition(s, a, r, s_)
            if step >= mem_size:
                if not start_train:
                    start_train = True
                    break
                rl.learn()
            else:
                human_transition.append((s, a, r, s_))
            s = s_
            step += 1
            # if start_train:
            #     if not step % 500: print('Timestep {} get score: {}'.format(step, sum(total_score[-500:])))
            #     total_score.append(performance)
            # if step == iteration:
            #     break;
        if start_train and step > mem_size:
            print('Episode {} get score: {}'.format(episode, performance))
            total_score.append(performance)
            episode += 1

    return total_score, human_transition

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        default='100',
                        dest='ITERATION',
                        help='input the iteration of training')

    parser.add_argument('-m',
                        default='10000',
                        dest='MEMORYSIZE',
                        help='input the size of memory')

    parser.add_argument('-b',
                        default='32',
                        dest='BATCHSIZE',
                        help='input the size of batch')

    parser.add_argument('-hu',
                        default='',
                        dest='HUMANEXP',
                        help='input human experience')

    args = parser.parse_args()
    
    try:
        iteration   = int(args.ITERATION)
        mem_size    = int(args.MEMORYSIZE)
        batch_size  = int(args.BATCHSIZE)
    except ValueError:
        print('error: iteration or memory size must be an integer')
        sys.exit()

    # game = gym.make('CartPole-v0')
    game = gym.make('MountainCar-v0')
    game = game.unwrapped

    rl_prioritized = RL(game.observation_space.shape[0] , range(game.action_space.n), batch_size=batch_size, memory_size=mem_size, prior=True, verbose=False)
    rl_dqn         = RL(game.observation_space.shape[0] , range(game.action_space.n), batch_size=batch_size, memory_size=mem_size, prior=False, verbose=False)

    import pickle

    if args.HUMANEXP == '':
        human_transition = None
    else:
        with open(args.HUMANEXP, 'rb') as f:
            human_transition = pickle.load(f)

    score_a, human_transition = play(game, rl_prioritized, iteration, human_transition)
    score_b, _                = play(game, rl_dqn, iteration, human_transition)

    with open('human_experience.pkl', 'wb') as f:
        pickle.dump(human_transition, f, -1)

    with open('score.pkl', 'wb') as f:
        pickle.dump({'a': score_a, 'b': score_b}, f, -1)
    
    # score_a = [ sum(score_a[i: i + 500]) for i in range(len(score_a) - 499)]
    # score_b = [ sum(score_b[i: i + 500]) for i in range(len(score_b) - 499)]


    plt.plot(range(len(score_a)), score_a, c='r', label='DQN with prioritized replay')
    plt.plot(range(len(score_b)), score_b, c='b', label='DQN')

    plt.show()
