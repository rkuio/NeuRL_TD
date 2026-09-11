import numpy as np 
import matplotlib.pyplot as plt



trials = 12
time_reward = 8 # less than trials
size_reward = 1.0 # if reward is present

learning_rate = 0.15


def run_trial(v, time_reward, size_reward, learning_rate, randoms, omit_reward=False, learn=True):
    random_time_reward = time_reward + randoms.integers(-1, 2)
    all_delta = np.zeros(trials + 1)
    for t in range(trials + 1):
        if t == random_time_reward and not omit_reward:
            r = size_reward
        else:
            r = 0 

        delta = r + v[t + 1] - v[t]
        all_delta[t] = delta 

        if learn and t > 0:
            v[t] = v[t] + learning_rate * delta # scaling it down then multiplying by reward 
    return all_delta 


inputs = [0, 1, 2, 3, 4]

all_first_traces = []
all_last_traces = []
all_omit_traces = []

for input in inputs:
    v = np.zeros(trials + 2) # all zeros random starting point
    randoms = np.random.default_rng(input)

    
    first_trace = run_trial(v, time_reward, size_reward, learning_rate, randoms)  # check deltas from previos run/ first trial 

    for i in range(200):
        run_trial(v, time_reward, size_reward, learning_rate, randoms)

    last_trace = run_trial(v, time_reward, size_reward, learning_rate, randoms, omit_reward=False, learn=False) # checked trials after the trial is over

    omit_trace = run_trial(v, time_reward, size_reward, learning_rate, randoms, omit_reward=True, learn=False) # going below baseline

    all_first_traces.append(first_trace)
    all_last_traces.append(last_trace)
    all_omit_traces.append(omit_trace)
    
print(first_trace)
print(last_trace)
print(omit_trace)

fig, axes = plt.subplots(3, 1, figsize=(7, 8))

time = range(len(all_first_traces[0]))

for trace in all_first_traces:
    axes[0].plot(time, trace, alpha=0.6)
axes[0].set_title("Early training")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("TD error (delta)")
axes[0].axvline(time_reward, color='green', linestyle='--', label='reward')

for trace in all_last_traces:
    axes[1].plot(time, trace, alpha=0.6)
axes[1].set_title("After learning")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("TD error (delta)")
axes[1].axvline(0, color='blue', linestyle='--', label='cue')
axes[1].axvline(time_reward, color='green', linestyle='--')

for trace in all_omit_traces:
    axes[2].plot(time, trace, alpha=0.6)
axes[2].set_title("Reward omitted")
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("TD error (delta)")
axes[2].axvline(0, color='blue', linestyle='--')
axes[2].axvline(time_reward, color='green', linestyle='--')

plt.tight_layout()
plt.savefig("td_dopamine_figure.png")
plt.show()



