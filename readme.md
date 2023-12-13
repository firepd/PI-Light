## $\pi$-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control AAAI 2024

code for $\pi$-Light.

## Dependencies

- python=3.8.10
- torch=1.7.1+cu110
- numpy=1.21.2
- CityFlow=0.1.0 

You need to install a modified version of [CityFlow](https://github.com/dxing-cs/TinyLight#dependencies) to run the code.

Then you need to unzip the data file.



### Run $\pi$-Light

```shell
python 02_run_MCTS.py --dataset=Jinan
```

### Evaluate generalization performance of $\pi$-Light

```shell
python 02_run_MCTS.py --dataset=Hangzhou1 --generalization=True target=Manhattan
```

### Run Tinylight

```shell
python 00_run_tiny_light.py --dataset=Jinan
```

### Run other baselines

```shell
python 01_run_baseline.py --dataset=Jinan
```

### Evaluate generalization performance of other baselines

```shell
python 015_baseline_transfer.py --dataset=Jinan
```

### Run [VIPER](https://arxiv.org/abs/1805.08328)

We also compare imitation learning-based VIPER, which distills the neural policy into a decision tree. We utilized MPLight as a teacher to generate state-action pairs for training the decision tree.
Overall, VIPER's performance is close to that of MPlight.

```shell
python 03_run_viper.py
```



## Acknowledgments

This codebase is based on [Tinylight](https://github.com/dxing-cs/TinyLight)'s code.