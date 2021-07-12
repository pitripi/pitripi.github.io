---
layout: post
title: "Training Policy Gradients"
description: "This post focuses on results of variance reduction techniques viz. batch sizes, reward-to-go, discounted returns, baselines and advantage normalization for policy gradient algorithms on CartPole-v0 environment."
date: 2019-06-05 20:17:37 +0530
image: '/images/policy_gradients/policy_gradients_title.jpeg'
tags:   [tech, reinforcement-learning, policy-gradients]
---

### __Notations__

| Symbol        |   Meaning           |
| ------------- |:-------------|
| $$s_t$$       | state at timestep $$t$$ |
| $$a_t$$       | action taken in state $$s_t$$ |
| $$r(a_t \vert s_t)$$ | reward after taking action $$a_t$$ in state $$s_t$$|
| $$\tau$$      | a trajectory, $$(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)$$ |
| $$r(\tau)$$   | $$\sum_{t}r(a_{t}, s_{t})$$ |
| $$\pi_\theta(\tau)$$ | probability of tajectory $$\tau$$ under policy $$\pi_\theta$$ with parameters $$\theta$$ |
| $$\gamma$$    | discount factor |
|  |  |
{: rules="groups"}

### __Theory__
Policy gradient methods try to solve reinforcement learning problems by modelling the policy as a parameterized function, generally a neural network. The policy network takes in observations as inputs and returns a distribution over available actions as output. An action from this distribution is sampled and a reward is received after the action is taken. The goal of policy gradient methods is to maximize (or minimize the negative of) an objective function which depends on the actions taken and rewards received over multiple timesteps. Our objective function, $$J(θ)$$ is the expectation of returns over all trajectories under policy $$\pi_\theta$$. We find the gradient of the objective functions w.r.t. the parameters of policy network and make updates.

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \bigg[ \sum_{t}r(a_{t}, s_{t})\bigg] \tag{1}\label{eq:1}$$

where,

$$
\begin{align}
  \pi_\theta(\tau) &= p(s_1)\sum_{t=1}^{T} \pi_\theta(a_t \vert s_t)p(s_{t+1} \vert s_t, a_T)
  \\ \log \pi_\theta(\tau) &= \log p(s_1) + \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) + \log p(s_{t+1} \vert s_t, a_T)
\end{align}
$$

The gradient of $$J(\theta)$$ has the following form.

$$
\begin{align}
    \nabla_\theta J(\theta)  &= \int \nabla_\theta \pi_\theta(\tau)r(\tau)d(\tau) 
  \\ &= \int \pi_\theta(\tau) \nabla_\theta \log\pi_\theta(\tau)r(\tau)d(\tau)
  \\ &= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \big[ \nabla_\theta \log\pi_\theta(\tau)r(\tau) \big]
  \\ &= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \bigg[ \bigg(\sum_{t=1}^{T} \nabla_\theta\log\pi_\theta(a_t \vert s_t)\bigg) r(\tau) \bigg]
  \\ &= \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \bigg[ \bigg( \sum_{t=1}^{T} \nabla_\theta\log\pi_\theta(a_t \vert s_t) \bigg) \sum_{t}r(a_{t}, s_{t}) \bigg]\tag{2}\label{eq:2}
\end{align}
$$

Computing this expectation under a distribution over all trajectories is intractable, so we use sampling instead to make an approximation.

$$\nabla_\theta J(\theta) \approx \frac {1}{N} \sum_{i=1}^{N} \bigg[ \bigg( \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) \bigg) \sum_{t}r(a_{t}, s_{t}) \bigg]\tag{3}\label{eq:3}$$

A simple policy gradient algorithm repeats the following steps:
1. Sample $$\tau^{i}$$ from $$\pi_\theta(a_t \vert s_t)$$
2. Compute $$\nabla_\theta J(\theta) \approx \frac {1}{N} \sum_{i} \big( ( \sum_{t} \nabla_\theta\log\pi_\theta(a_t \vert s_t) ) \sum_{t=1}^{T}r(a_{it}, s_{it}) \big)$$  
3. Update $$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

This simple algorithm is prone to variance. While training policy gradients, there are certain methods used to reduce such variance. These methods usually make small changes in the objective function. In the following sections, the effects of these methods is demonstrated on CartPole-v0 environment.

__Practical consideration:__ Computing $$\nabla_\theta\log\pi_\theta(a_t \vert s_t)$$ explicitly is inefficient. We instead compute 
$$\tilde{J}(\theta) \approx \frac {1}{N} \sum_{i=1}^{N} \bigg[ \bigg( \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) \bigg) \sum_{t}r(a_{t}, s_{t}) \bigg]\tag{4}\label{eq:4}$$
and use automatic differentiation to compute policy gradients.

### __CartPole-v0__
[CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) is an OpenAI gym environment where a pole is attached to a cart. The cart can move along a frictionless track. The goal is to prevent the pole from falling.  CartPole-v0 observations are arrays of 4 values - cart position, cart velocity, pole angle and pole velocity at the top. The action is a discrete value - 0 (push cart to the left) or 1 (push cart to the right). The top limit on the number of steps that can occur in one episode is 200. There is another version, CartPole-v1 with 500 steps as max limit but experiments in this article will be based on CartPole-v0.

### __Experiment details__

* __Policy network:__ inputs of dimensions (batch size, 4); 2 hidden layers of size 64 each; output dimensions of (batch size, 2); learning rate of 0.001 with Adam optimizer
* __Baseline network:__ input of dimensions (batch size, 4); 2 hidden layers of size 64 each; output dimensions of (batch size, 1); learning rate of 0.001 with Adam optimizer
* __Discount factor:__ 0.99
* __Random seeds:__ All experiments are performed thrice with numpy, pytorch and gym seeds as 0, 10 and 20. Results are averaged.
* __Number of iterations:__ 1000. One iteration means using a batch to make one policy update.
* __Plots:__ Average of returns in a batch vs iterations. 

#### __Batch size__
Batch size refers to the number of timesteps used to make an update to policy network. When only a single timestep is used, the algorithm is called REINFORCE. Here, comparison is done between three batch sizes:

1. At least 1. Only a single episode is run.
2. At least 500. Episodes are run until we get 500 timesteps.
3. At least 1000. Episodes are run until we get 1000 timesteps.


It is possible that we receive required minimum timesteps (1, 500 or 1000) in the middle of an episode but we still run the episode till the end and use all the timesteps from that episode in the batch. This leads us to having some extra timsteps in our batch. If we stop the episode in the middle, then the returns we get will be lower than actual returns. Now, let's run our algorithm to maximize \eqref{eq:4}.

![2l_64s_1b_0na_0rtg_0nnb_0.99g_0.001lr]({{site.baseurl}}/images/policy_gradients/loss_sum/2l_64s_1b_0na_0rtg_0nnb_0.99g_0.001lr.jpg)
*Policy update using at least 1 timestep.*

We can see that average returns of the batches are not consistent and have a lot of variance. Also, notice that the results are sensitive to random seeds as well. On increasing the batch size to 500, we get a considerable reduction in variance and also, a faster increase in average returns.

![2l_64s_500b_0na_0rtg_0nnb_0.99g_0.001lr]({{site.baseurl}}/images/policy_gradients/loss_sum/2l_64s_500b_0na_0rtg_0nnb_0.99g_0.001lr.jpg)
*Policy update using at least 500 timesteps.*

There are still inconsistencies around higher rewards. Using a batch size of 1000 gives us a smoother curve. These results were observed without using discounted returns. Next, we'll change our objective function to use causality and discounted returns.

![2l_64s_1000b_0na_0rtg_0nnb_0.99g_0.001lr]({{site.baseurl}}/images/policy_gradients/loss_sum/2l_64s_1000b_0na_0rtg_0nnb_0.99g_0.001lr.jpg)
*Policy update using at least 1000 timesteps.*

#### __Causality and reward-to-go__
In the objective function used above, log probabilities of all taken actions are multiplied by sum of rewards. But policy at timestep $$t'$$ cannot affect reward at timestep $$t$$ when $$t < t'$$. So we modify the objective function to:

$$\tilde{J}(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \bigg(\sum_{t=1}^{T}\log\pi_\theta(a_{it}\vert s_{it})\bigg)\bigg(\sum_{t’=t}^{T} r(a_{it’}\vert s_{it’})\bigg)\tag{5}\label{eq:5}$$

Also, rewards received much later in future should have lesser contribution to returns than rewards received in near future. So, we use discounted returns to incorporate this dynamic behaviour.

$$\tilde{J}(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \bigg(\sum_{t=1}^{T}\log\pi_\theta(a_{it}\vert s_{it})\bigg)\bigg(\sum_{t’=t}^{T} \gamma^{t’-t} r(a_{it’}\vert s_{it’})\bigg)\tag{6}\label{eq:6}$$

On using causality and reward-to-go, our policy learns faster and reaches the maximum of average returns in fewer timesteps. 

![rtg_nortg_1]({{site.baseurl}}/images/policy_gradients/loss_sum/rtg_nortg_1.jpg)
*Reward-to-go (yellow) vs no reward-to-go (blue) using at least 1 timestep.*

![rtg_nortg_500]({{site.baseurl}}/images/policy_gradients/loss_sum/rtg_nortg_500.jpg)
*Reward-to-go (yellow) vs no reward-to-go (blue) using at least 500 timestep.*

![rtg_nortg_1000]({{site.baseurl}}/images/policy_gradients/loss_sum/rtg_nortg_1000.jpg)
*Reward-to-go (yellow) vs no reward-to-go (blue) using at least 1000 timestep.*

#### __Baselines__
A popular variation of the objective function to reduce variance of gradient function is to subtract a baseline from the returns. A common baseline to subtract is state-value function and the resulting value is called advantage. 

$$\tilde{J}(\theta) \approx \frac {1}{N} \sum_{i=1}^{N} \bigg[ \bigg( \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) \bigg) \bigg(\sum_{t'=t}^{T}r(a_{it}, s_{it}) - b\bigg) \bigg]\tag{7}\label{eq:7}$$

We now optimize \eqref{eq:7} using a neural network baseline 

![nnb_nonnb_1]({{site.baseurl}}/images/policy_gradients/loss_sum/nnb_nonnb_1.jpg)
*Baseline (yellow) vs no baseline (blue) using at least 1 timestep.*

![nnb_nonnb_500]({{site.baseurl}}/images/policy_gradients/loss_sum/nnb_nonnb_500.jpg)
*Baseline (yellow) vs no baseline (blue) using at least 500 timestep.*

![nnb_nonnb_1000]({{site.baseurl}}/images/policy_gradients/loss_sum/nnb_nonnb_1000.jpg)
*Baseline (yellow) vs no baseline (blue) using at least 1000 timestep.*

Training plots show that baseline method works better for large batch sizes. It does not seem to work with small batch sizes (or when our batch consists of only one episosde).

#### __Advantage normalization__
Another trick to make policy gradients work is normalizing returns or advantage (in case of baselines). Following are the results of normalizing returns, baselines were not used in this case.

![na_nona_1]({{site.baseurl}}/images/policy_gradients/loss_sum/na_nona_1.jpg)
*Normalization (yellow) vs no normalization (blue) using at least 1 timestep.*

![na_nona_500]({{site.baseurl}}/images/policy_gradients/loss_sum/na_nona_500.jpg)
*Normalization (yellow) vs no normalization (blue) using at least 500 timestep.*

![na_nona_1000]({{site.baseurl}}/images/policy_gradients/loss_sum/na_nona_1000.jpg)
*Normalization (yellow) vs no normalization (blue) using at least 1000 timestep.*


#### Final result
This time we use everything we have in our toolset - causality, reward-to-go, baselines and advantage normalization.

![final]({{site.baseurl}}/images/policy_gradients/loss_sum/final.jpg)
*Using causality, reward-to-go, baselines and advantage normalization.*


You can see that our small batch size is performing poorly due to added baseline.

Is there a fix to make baseline work with smaller batches? So far, we have used the objective function of the form 

$$\tilde{J}(\theta) \approx \frac {1}{N} \sum_{i=1}^{N} \big[ \big( \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) \big) \sum_{t}r(a_{t}, s_{t}) \big]$$ 

which can be thought of as an average of returns over sampled trajectories. We can instead use average of returns over all timesteps i.e.

$$\tilde{J}(\theta) \approx \frac {1}{\tilde{T}} \sum_{i=1}^{N} \bigg[ \bigg( \sum_{t=1}^{T} \log\pi_\theta(a_t \vert s_t) \bigg) \sum_{t}r(a_{t}, s_{t}) \bigg]\tag{8}\label{eq:8}$$

where $$\tilde{T}$$ is total timesteps in a batch. Optimizing \eqref{eq:8} fixes the baseline inconsistency for small batches.

![final_mean]({{site.baseurl}}/images/policy_gradients/loss_mean/final.jpg)
*Using average of returns over all timesteps.*

### __References__
1. [http://rail.eecs.berkeley.edu/deeprlcourse/](Berkeley Deep RL Course)
2. [https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](Policy Gradient Algorithms)
3. [https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/](Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients)