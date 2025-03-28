# A/B Testing Project

## Overview

This repository contains an A/B testing experiment using multi-armed bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to optimize advertisement selection by maximizing cumulative rewards and minimizing regret.

## Structure

- `csv/`: Stores the generated output files (`epsilon_rewards.csv`, `thompson_rewards.csv`)
- `Bandit.py`: Main script that implements the bandit algorithms, runs experiments, logs results, and generates visualizations
- `requirements.txt`: Lists required Python libraries

## Requirements

Install all dependencies by running:

```bash
pip install -r requirements.txt


