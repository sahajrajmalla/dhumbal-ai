# Dhumbal AI: Comparative Study of AI Agents

## Overview

This repository implements AI agents for the Dhumbal card game, a South Asian multiplayer game with imperfect information. The project, detailed in *"AI Agents for the Dhumbal Card Game: A Comparative Study"*, evaluates rule-based (Aggressive, Conservative, Balanced, Opportunistic), search-based (MCTS, ISMCTS), learning-based (DQN, PPO), and random agents through 1024-round tournaments. It includes game environment, agent implementations, statistical analysis, and visualizations, supporting game AI research and cultural preservation.

## Features

- **Game Environment**: Python-based Dhumbal implementation (2–5 players).
- **AI Agents**: Random, rule-based, MCTS, ISMCTS, DQN, PPO.
- **Evaluation**: Win rate, economic performance, Jhyap success, decision efficiency.
- **Analysis**: Welch’s t-tests, Cohen’s d, 95% CI, visualized via Jupyter notebooks.
- **Reproducibility**: Fixed random seed (42), Python 3.9+, TensorFlow 2.8.

## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/sahajrajmalla/dhumbal-ai.git
   cd dhumbal-ai
   ```

2. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Quick Demo**:
   ```bash
   python quick_demo.py
   ```

- **Tournaments**:
  - Rule-Based: `python agents/rule_based/rule_based_agent.py`
  - Search-Based: `python agents/search_based/single_process_search.py`
  - Learning-Based: `python agents/learning_based/compete.py`
  - Championship: `python championship/final.py`

- **Visualizations**:
   ```bash
   cd visualization
   jupyter notebook
   ```
   Run `performance_metrics.ipynb` and `statistical_analysis.ipynb`.

- **Training**:
  - DQN: `python agents/learning_based/dqn/dqn.py`
  - PPO: `python agents/learning_based/ppo/ppo.py`

## Repository Structure

```
dhumbal-ai/
├── agents/
│   ├── rule_based/           # Rule-based agents and results
│   ├── search_based/         # MCTS, ISMCTS, and results
│   ├── learning_based/       # DQN, PPO, training, and results
│   └── hybrid/               # Placeholder for hybrid agents
├── championship/             # Cross-category tournament scripts and results
├── visualization/            # Notebooks and plots for analysis
├── quick_demo.py             # Demo script
├── README.md                 # This file
├── LICENSE                   # MIT License
└── .gitignore
```

## Contributing

Fork, create a branch, commit changes, and submit a pull request. Follow PEP 8 and document code.

## License

MIT License (see [LICENSE](LICENSE)).

## Citation

<!-- ```bibtex
@article{malla2025dhumbal,
  author={Malla, Sahaj Raj},
  title={AI Agents for the Dhumbal Card Game: A Comparative Study},
  journal={IEEE Transactions on Games},
  year={2025}
}
``` -->

## Contact

Sahaj Raj Malla: [sm03200822@student.ku.edu.np](mailto:sm03200822@student.ku.edu.np)