## $\pi$-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control

code for $\pi$-Light.

## Dependencies

- python=3.8.10
- torch=1.7.1+cu110
- numpy=1.21.2
- CityFlow=0.1.0 



### Run $\pi$-Light

```shell
python 02_run_MCTS.py --dataset=Jinan
```

#### Evaluate generalization performance of $\pi$-Light

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



## Acknowledgments

This codebase is based on Tinylight's code.