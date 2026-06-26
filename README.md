# Representation Learning Enables Scalable Multitask Deep Reinforcement Learning

Official code repository for the paper:

**[Representation Learning Enables Scalable Multitask Deep Reinforcement Learning](https://arxiv.org/abs/2606.05555)**

[Johan Obando-Ceron](https://johanobandoc.github.io), [Lu Li](https://mila.quebec/en/directory/lu-li), [Scott Fujimoto](https://scholar.google.com/citations?user=1Nk3WZoAAAAJ&hl=en), [Pierre-Luc Bacon](https://pierrelucbacon.com), [Aaron Courville](https://mila.quebec/en/directory/aaron-courville), [Pablo Samuel Castro](https://psc-g.github.io)

This repository builds on [MMBench & Newt](https://github.com/nicklashansen/newt) — [Learning Massively Multitask World Models for Continuous Control](https://arxiv.org/abs/2511.19584).

----

## Running agents

Available task sets: `dmcontrol`, `dmcontrol-ext`, `metaworld`, `mujoco`, `box2d`, `robodesk`, `ogbench`, `pygame`, `atari`, `maniskill`

### agent=mrq

```bash
python tdmpc2/train.py use_demos=false agent=mrq task=<TASK_SET> seed=<SEED> steps=10000000 horizon=5
```

### agent=tdmpc2

```bash
python tdmpc2/train.py use_demos=false agent=tdmpc2 task=<TASK_SET> seed=<SEED> steps=10000000
```

----

## Citation

If you find this work useful, please cite:

```bibtex
@misc{obandoceron2026representationlearningenablesscalable,
      title={Representation Learning Enables Scalable Multitask Deep Reinforcement Learning},
      author={Johan Obando-Ceron and Lu Li and Scott Fujimoto and Pierre-Luc Bacon and Aaron Courville and Pablo Samuel Castro},
      year={2026},
      eprint={2606.05555},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.05555},
}
```

If you use the MMBench environment suite or the Newt baseline, please also cite:

```bibtex
@misc{Hansen2025Newt,
      title={Learning Massively Multitask World Models for Continuous Control},
      author={Nicklas Hansen and Hao Su and Xiaolong Wang},
      year={2025},
      eprint={2511.19584},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.19584},
}
```

----

## Contributing

We welcome contributions focused on improving sample efficiency and wall clock time performance in multitask reinforcement learning. If you have a proposal for a more efficient training component or want to add support for a new task, please open an issue or pull request. We are particularly interested in contributions that push the boundaries of scalability across diverse task sets.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
