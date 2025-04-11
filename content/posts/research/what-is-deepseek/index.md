---
title: "DeepSeek: A Deep Dive into the Next Generation Language Model"
date: "2024-03-20"
description: "An in-depth analysis of DeepSeek, exploring its architecture, capabilities, and how it compares to other leading language models in the field of AI"
excerpt: "Discover how DeepSeek is pushing the boundaries of language models with its innovative architecture and impressive performance metrics"
category: "Research"
tags: ["artificial-intelligence", "language-models", "deep-learning", "nlp", "deepseek", "machine-learning"]
author: "Sajjad Momeni"
readTime: "10 min read"
hero: "./hero.png"
--- 

- chain of thought
what are good criteria for a good chain of thought?
- correct CoT
- continuous reasoning -> easy to understand, no hidden states
- contain trial and error

how to find chain of thought?
- internet? no
- textbooks? no
- papers? no
- ...

data is the fossil fuel of AI
We've achieved peak of data and there is no more data to be collected


DeepSeek R1 is important not just because of what it can do, but because of how it does it.

Chain of Thought makes AI more transparent.
Reinforcement learning makes it more self-improving.
Distillation makes it more available.

I want to explain each of these techniques in detail.


1. Chain of Thought: prompt += "Let's think step by step"
2. Reinforcement Learning: reward function
<!-- add a gif from a baby learning to walk -->
![baby learning to walk](baby-learning-to-walk.gif)

3. Distillation:
We have to model the teacher model and then use it to train the student model.
The teacher model is a large language model that has been trained on a large dataset.
The student model is a smaller language model that has been trained on a smaller dataset which this dataset is the output of the teacher model. 
![Knowledge Distillation Process](https://miro.medium.com/max/1400/1*9uNYZMG3RqFGJkKG7Z4ECQ.png)
*Knowledge Distillation Process: Teacher model transfers knowledge to a smaller student model*

##Prompt tokens
- <think></think>
- <answer></answer>

<!-- write a code section for train the model -->
for a question q, a group of responses {o1, o2, o3, o4, o5} received with rewards {r1, r2, r3, r4, r5} of {1, 0, 0, 1, 1} respectively.
1. Compute the advantage of each response:
<!-- code section -->
```python
# Responses and their rewards
responses = ['o1', 'o2', 'o3', 'o4', 'o5']
rewards = [1, 0, 0, 1, 1]

# Calculate mean reward (baseline)
baseline = sum(rewards) / len(rewards)

# Calculate advantage for each response
# Advantage = reward - baseline
advantages = []
for reward in rewards:
    advantage = reward - baseline
    advantages.append(advantage)

# Print results
for i, (response, reward, advantage) in enumerate(zip(responses, rewards, advantages)):
    print(f"Response {response}: Reward = {reward}, Advantage = {advantage:.2f}")
# Response o1: Reward = 1, Advantage = 0.40
# Response o2: Reward = 0, Advantage = -0.60
# Response o3: Reward = 0, Advantage = -0.60
# Response o4: Reward = 1, Advantage = 0.40
# Response o5: Reward = 1, Advantage = 0.40

```

2. Compute the probability of each response:
```python

```



Further Reading: 
- [Reinforcement Learning](/posts/research/reinforcement-learning)
- [AI Agents and how to build them](/posts/research/ai-agents)




