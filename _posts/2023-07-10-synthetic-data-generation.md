---
layout: post
title:  "Scalable modular synthetic data generation for advancing aerial autonomy"
date:   2023-07-11
desc: "Synthetic data generation for Autonomous Systems"
keywords: "synthetic data, autonomous systems, aerial mobility, machine learning"
categories: [Autonomous-systems]
tags: [sakshi-mishra, autonomous-systems]
icon: icon-html
---

## Machine Learning models: required datasets
To advance autonomy in vision-based navigation and detection, Machine Learning (ML) models require a vast number of aerial datasets for training. But real-life data collection through deploying drones and iteratively testing models and algorithms based on that data in various preliminaries is tedious, expensive, and limited. This has led to increased reliance on synthetic data.

Existing methods for creating aerial synthetic data are based on limited training simulation scenes, which can cause the model to overfit and, as a result, make it difficult for the model to generalize to unknown environments. The reality gap (sim-to-real) between the simulation and the real world can widen as a result of a lack of diversity in the scenes, which can result in performance issues when transferring models due to distribution differences. Models must be trained over a diverse set of datasets in a variety of simulation environments in order to improve their generalizability to various real-world domains and circumstances.

## Challenges with existing synthetic data generation tools
The current synthetic data generation tools have limitations, such as the lack of data augmentation or heavy reliance on manual work and real samples for generating realistic simulation scenes. These limitations restrict the scalability of the data generation workflow and create a challenge in balancing generalizability and scalability. 

---

**NOTE**:
The process of effective large-scale dataset generation needs two key elements in place - generalizability and scalability of the mechanism. 

---

This exactly is a very real challenge in the industry today.

So then what are the possible solutions to these challenges?

### Possible solutions
There are a couple of solutions that are being experimented with or in some cases implemented to overcome the limitations of the current synthetic data generation tools. It has been demonstrated that increasing the diversity of simulation environments used to train a model across all varieties or augmenting the data through Domain Randomization (DR), is very effective in addressing this issue. For large-scale, diverse scene generation, current synthetic data generation techniques either lack integrated data augmentation workflows or increasingly rely on manual tasks for setting simulation parameters and designing simulation scenes which require very specialized knowledge about the simulation software and are not scalable.

## Our recent work
To address these gaps, we introduce a scalable Aerial Synthetic Data Augmentation (ASDA) framework tailored to aerial autonomy applications. The procedural generative approach of our data augmentation is performant and adaptable to different simulation environments, training tasks, and data collection needs. We demonstrate the effectiveness of our method in automatically generating diverse datasets and show its potential for downstream performance optimization.

Our work contributes to generating enhanced benchmark datasets for training models that can generalize better to real-world situations.
- Link to [arXiv preprint](https://arxiv.org/ftp/arxiv/papers/2211/2211.05335.pdf)
- Link to [summary video](youtube.com/watch?v=eKpOh-K-NfQ)
- Link to Robotics and Autonomous Systems journal [publication](https://www.sciencedirect.com/science/article/abs/pii/S0921889023001033)
