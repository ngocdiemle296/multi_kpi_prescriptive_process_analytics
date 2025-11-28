# Balancing KPIs through Multi-Objective Precriptive Process Analytics

This repository contains the implementation of the multi-KPI optimization framework for prescriptive process analytics, as described in our paper **Balancing KPIs through Multi-Objective Precriptive Process Analytics**.

## Abstract
Prescriptive Process Analytics aims to support ongoing process executions by guiding them toward desirable outcomes, which are usually measured through Key Performance Indicators (KPIs). While the literature on Prescriptive Process Analytics has focused on the exploration of different recommendation techniques and tasks, the current efforts have so far largely focused on optimizing a single KPI. However, real-world processes often involve multiple, interrelated, and potentially conflicting KPIs, requiring an explicit treatment of their trade-offs. This paper presents a framework that generates recommendations designed to optimize multiple KPIs simultaneously, and that allows users to explore different levels of trade-off among them. Our proposed framework jointly recommends the next activity and its executing resource, which we call next step recommendation, aiming to balance multiple KPIs. Different strategies are proposed to generate these recommendations, among which one using a genetic algorithm that efficiently explores the search space of viable options. This is particularly effective in domains with a large number of resources, which lead to a significantly extensive space of possible recommendations.
Experimental results demonstrate that our multi-KPI approach achieves favorable trade-offs among partially conflicting goals, while minimizing the degradation of each individual KPI compared to single-objective optimization. 


## Framework
<img width="4536" height="2914" alt="framework_reduced_size" src="https://github.com/user-attachments/assets/86c1ea1f-231b-4058-81bf-0725fdfa9489" />


## Installation
**Dependencies**

This implementation requires the following Python libraries:
```python
pip install pandas numpy catboost dice_ml
```

## Usage
1. Preprocessing Event Logs
2. Train the Predictive model
3. Generate Recommendations
```python
python run_experiment.py --case_study "bac" --reduced_threshold 0.5 --num_cfes 5
```

## License
This project is licensed under the MIT License.
