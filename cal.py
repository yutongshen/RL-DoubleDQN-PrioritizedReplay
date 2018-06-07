import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def errorfill(x, y, yerr, color, alpha_fill=0.3, label=None, ax=None):
    ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, label=label)
    ax.margins(y=0.10)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-n',
                        default='10',
                        dest='NUM',
                        help='input data size')

    parser.add_argument('-t',
                        default='b32',
                        dest='MODELTYPE',
                        help='input the type of the model')

    parser.add_argument('-e',
                        default='100',
                        dest='EPISODE',
                        help='input the number of episode')

    
    args = parser.parse_args()
    summary = {}
    model_type = args.MODELTYPE

    try:
        n = int(args.NUM)
    except ValueError:
        print('Data size must be an integer')
        sys.exit()

    try:
        episode = int(args.EPISODE)
    except ValueError:
        print('Episode must be an integer')
        sys.exit()

    total_score = np.zeros((2, episode, n), np.float32)
    summary[model_type] = np.zeros((2, 2, episode), np.float32)
    
    for i in range(n):
        score_file = 'score/score_' + model_type + '_' + str(i).zfill(2) + '.pkl'
        with open(score_file, 'rb') as f:
            score  = pickle.load(f)
        for j in range(episode):
            total_score[0, :, i] = np.array(score['a'])
            total_score[1, :, i] = np.array(score['b'])
    
    summary[model_type][0, 0] = np.mean(total_score[0], axis=1)
    summary[model_type][0, 1] = np.std(total_score[0], axis=1)
    summary[model_type][1, 0] = np.mean(total_score[1], axis=1)
    summary[model_type][1, 1] = np.std(total_score[1], axis=1)
        
    errorfill(range(len(summary[model_type][0, 0])), summary[model_type][0, 0], summary[model_type][0, 1], 'blue', label='prioritized')
    errorfill(range(len(summary[model_type][1, 0])), summary[model_type][1, 0], summary[model_type][1, 1], 'red', label='uniform sampling')
    
    blue_patch = mpatches.Patch(color='blue', label='prioritized')
    red_patch  = mpatches.Patch(color='red',  label='uniform sampling')
    plt.legend(handles=[blue_patch, red_patch])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    
    plt.show()


    # print(summary[model_type][0, 0])
    # print(summary[model_type][0, 1])
    # print(summary[model_type][1, 0])
    # print(summary[model_type][1, 1])
