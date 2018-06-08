# Reinforce Learning Practice
Mountain Car (OpenAI gym) - Double Deep Q Network with Prioritized Experience Replay

## Prerequisite
- Python 3.6.4

## Install Dependency
```sh
$ pip install -r requirements.txt
```

## Usage
```sh
$ python usage: main.py [-h] [-i ITERATION] [-m MEMORYSIZE] [-b BATCHSIZE] [-lr LEARNINGRATE] [-hu HUMANEXP] [-hu--out HUMANEXPOUT] [-score--out SCOREOUT] [-screen SCREEN]
```

| optional Options           | Description                                    |
| ---                        | ---                                            |
| -h, --help                 | show this help message and exit                |
| -i ITERATION               | input the iteration of training                |
| -m MEMORYSIZE              | input the size of memory                       |
| -b BATCHSIZE               | input the size of batch                        |
| -lr LEARNINGRATE           | input learning rate                            |
| -hu HUMANEXP               | input human experience                         |
| -hu--out HUMANEXPOUT       | human experience output path                   |
| -score--out SCOREOUT       | score output path                              |
| -screen SCREEN             | show the screen of game (true/false)           | 


## Game
![MountainCar](https://user-images.githubusercontent.com/8510097/31701297-3ebf291c-b384-11e7-8289-24f1d392fb48.PNG)

## Algorithm
- Double Deep Q Network With Prioritized Experience Replay
  - **Input:** minibatch k, step-size η, replay period K and size N, exponents α and β, budget
  - Initialize replay memory H = Φ, ![\Delta](https://latex.codecogs.com/svg.latex?\Delta) = 0, ![p_1](https://latex.codecogs.com/svg.latex?p_1) = 1
  - Observe ![S_0](https://latex.codecogs.com/svg.latex?S_0) and choose ![A_0\sim\pi_\theta(S_0)](https://latex.codecogs.com/svg.latex?A_0\sim\pi_\theta%28S_0%29)
  - **for** t = 1 **to** T **do**
    - Observe ![S_t,R_t,\gamma_t](https://latex.codecogs.com/svg.latex?S_t,R_t,\gamma_t)
    - Store transition ![(S_t−1, A_t−1, R_t, \gamma_t, S_t)](https://latex.codecogs.com/svg.latex?%28S_{t−1}, A_{t−1}, R_t, \gamma_t, S_t%29) in H with maximal priority ![pt=maxitpi](https://latex.codecogs.com/svg.latex?p_t=\max_{i<t}p_i)

  - Initialize Q network with parameters θ
  - Initialize enviroment and get current state s
  - According to s, Actor will give an action a: (ε-Greedy, e.g. ε = 0.9)
    - 10%: random choose one of actions 
    - 90%: choose the action with the highest ![Q(s:\theta)](https://latex.codecogs.com/svg.latex?Q%28s;\theta%29)
  - Take the action, and observe the reward, r, as well as the new state, s'.
  - Update the θ for the state using the observed reward and the maximum reward possible for the next state.
    - ![L=(r+\gammaQ(s',argmax\_{a'}Q(s',a';\theta);\theta^{-})-Q(s,a:\theta))^{2}](https://latex.codecogs.com/svg.latex?L=%28r+\gamma%20Q%28s',argmax_{a%27}Q%28s%27,a%27;\theta%29;\theta^{-}%29-Q%28s,a;\theta%29%29^{2})
    - ![\theta=\theta-lr\triangledown\_\thetaL](https://latex.codecogs.com/svg.latex?\theta=\theta-lr\triangledown_\theta%20L)
  - Every C steps reset ![\theta^{-}\leftarrow\theta](https://latex.codecogs.com/svg.latex?\theta^{-}\leftarrow\theta)
  - Set the state to the new state, and repeat the process until a terminal state is reached.

## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)
