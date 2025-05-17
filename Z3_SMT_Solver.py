import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from z3 import *

# Define the project activities and their durations (mean and standard deviation)
activities = {
    'A': (2, 0),
    'B': (1, 0.2), 'C': (3, 0.7), 'D': (4, 1.0), 'E': (2, 0.5),
    'F': (5, 1.2), 'G': (2, 0.4), 'H': (1, 0.2), 'I': (2, 0.5), 'J': (1, 0.2), 'K': (2, 0.5)
}

# Define the dependencies between activities
dependencies = {
    'A': [], 'B': ['A'], 'C': ['B'], 'D': ['B'], 'E': ['B'], 'F': ['E'],
    'G': ['C'], 'H': ['C', 'D'], 'I': ['F', 'G'], 'J': ['H', 'I'], 'K': ['J']
}

# --- Define Custom Duration Functions (Example) ---
def dur_normal(mean, std):
    """Returns a duration based on a normal distribution, ensure value >= 1"""
    return max(1, round(np.random.normal(mean, std), 0))

def durA():
    return 4

def durB():
    """Simulate duration for task B"""
    avg = 60
    stdev = 10
    component_availability = max(50, np.around(np.random.normal(avg, stdev)))
    if component_availability >= 80:
        return 3
    else:
        return round(5 - (5 - 3) / (80 - 50) * (component_availability - 50))

def durC():
    tC_dur = [2, 3, 4]
    tC_prob = [0.3, 0.5, 0.2]
    return np.random.choice(tC_dur, 1, p=tC_prob)[0]

def durD():
    quality = 10 * np.random.uniform()
    return round(8 - (8 - 4) / 10 * quality)

# --- Updated CPM Logic ---
def init_nodes():
    nodes = {activity: {'dur': 0, 'ES': 0, 'EF': 0, 'LS': float('inf'), 'LF': float('inf'),
                        'float': 0, 'drag': 0, 'pred': [], 'succ': [], 'CP': False, 'n_in_CP': 0}
             for activity in activities}
    for activity, preds in dependencies.items():
        nodes[activity]['pred'] = preds.copy()
        for pred in preds:
            if activity not in nodes[pred]['succ']:
                nodes[pred]['succ'].append(activity)
    return nodes

def forward_pass(nodes):
    """Perform the forward pass to calculate ES and EF for each activity."""
    # Reset ES and EF values
    for node in nodes.values():
        node['ES'] = 0
        node['EF'] = 0

    processed = set()
    while len(processed) < len(nodes):
        for activity, data in nodes.items():
            if activity in processed:
                continue

            # Process if it's a start activity or all predecessors are processed
            if not data['pred'] or all(pred in processed for pred in data['pred']):
                if data['pred']:
                    data['ES'] = max(nodes[pred]['EF'] for pred in data['pred'])
                data['EF'] = round(data['ES'] + data['dur'], 3)
                processed.add(activity)

def backward_pass(nodes):
    """Perform the backward pass to calculate LS and LF for each activity."""
    # Reset LS and LF values
    for node in nodes.values():
        node['LS'] = float('inf')
        node['LF'] = float('inf')

    # Find project completion time
    project_duration = max(node['EF'] for node in nodes.values())

    # Process in reverse order
    processed = set()
    while len(processed) < len(nodes):
        for activity, data in nodes.items():
            if activity in processed:
                continue

            # Process if it's an end activity or all successors are processed
            if not data['succ'] or all(succ in processed for succ in data['succ']):
                if not data['succ']:
                    data['LF'] = project_duration
                else:
                    data['LF'] = min(nodes[succ]['LS'] for succ in data['succ'])
                data['LS'] = round(data['LF'] - data['dur'], 3)
                processed.add(activity)

def calculate_critical_path(nodes):
    """Calculate the critical path by performing forward and backward passes."""
    forward_pass(nodes)
    backward_pass(nodes)

    # Calculate float and identify critical path
    for activity, data in nodes.items():
        data['float'] = round(data['LS'] - data['ES'], 3)
        data['CP'] = abs(data['float']) < 0.001  # Using small threshold for float comparison

def calculate_drag(nodes):
    """Calculate the drag for each activity on the critical path."""
    # Store original project duration
    forward_pass(nodes)
    backward_pass(nodes)
    original_duration = max(node['EF'] for node in nodes.values())

    # Calculate drag for each activity
    for activity, data in nodes.items():
        if not data['CP']:
            data['drag'] = 0
            continue

        # Store original duration and temporarily set to zero
        original_act_duration = data['dur']
        data['dur'] = 0

        # Recalculate project duration with activity duration = 0
        forward_pass(nodes)
        backward_pass(nodes)
        new_duration = max(node['EF'] for node in nodes.values())

        # Calculate drag
        data['drag'] = round(original_duration - new_duration, 3)

        # Restore original duration
        data['dur'] = original_act_duration

    # Final calculations to restore all values
    forward_pass(nodes)
    backward_pass(nodes)

    # --- Simulation ---
reps = 1000
proj_durations = []
all_nodes = init_nodes()  # for storing results

for i in range(reps):
    nodes = init_nodes()
    # Sample durations
    for activity in activities:
        if activity == 'A':
            nodes[activity]['dur'] = durA()
        elif activity == 'B':
            nodes[activity]['dur'] = durB()
        elif activity == 'C':
            nodes[activity]['dur'] = durC()
        elif activity == 'D':
            nodes[activity]['dur'] = durD()
        else:
            nodes[activity]['dur'] = dur_normal(activities[activity][0], activities[activity][1])

    calculate_critical_path(nodes)
    # Keep track of how many times the task appeared on the critical path
    for activity, data in nodes.items():
        if data['CP']:
            all_nodes[activity]['n_in_CP'] += 1
    # Record the project duration
    proj_durations.append(max(node['EF'] for node in nodes.values()))

# Calculate mean project duration
mean_project_duration = np.mean(proj_durations)

# Recalculate drag on mean project duration
for activity in all_nodes:
    all_nodes[activity]['dur'] = activities[activity][0]
calculate_critical_path(all_nodes)
calculate_drag(all_nodes)

# --- Z3 Optimization ---
opt = Optimize()
project_duration = Real('project_duration')
opt.add(project_duration == mean_project_duration)

# Add constraints for each activity
activity_vars = {activity: Real(activity) for activity in activities}
for activity, data in all_nodes.items():
    opt.add(activity_vars[activity] == data['dur'])

# Define the objective to minimize project duration
opt.minimize(project_duration)

# Check and extract the model
if opt.check() == sat:
    model = opt.model()
    optimized_durations = {activity: model[activity_vars[activity]].as_decimal(2) for activity in activities}
    print("Optimized Durations:")
    for activity, duration in optimized_durations.items():
        print(f"{activity}: {duration}")
else:
    print("No optimal solution found.")

# --- Output and Analysis ---
# Project duration histogram
pdf = pd.Series(proj_durations).groupby(pd.Series(proj_durations)).count()  # Series
pdf = pdf.reindex(range(int(pdf.index[0]), int(pdf.index[-1]) + 1), fill_value=0)

print("Project duration distribution:")
print(pdf.to_list())
plt.figure(figsize=(10, 6))
plt.bar(range(pdf.index[0], pdf.index[-1] + 1), pdf)
plt.title('Project Duration Distribution')
plt.xlabel('Project Duration')
plt.ylabel('Frequency')
plt.xticks(range(pdf.index[0], pdf.index[-1] + 1))
plt.show()

# Calculate mean project duration
mean_project_duration = np.mean(proj_durations)
print(f"Mean project duration: {mean_project_duration}")

# Calculate 95th percentile
percentile_95 = np.percentile(proj_durations, 95)
print(f"95th percentile project duration: {percentile_95}")

# Critical Path Frequency
print("\nCritical Path Frequency:")
df_CP_freq = pd.DataFrame([dict(Task=key, Freq=int(val['n_in_CP'] / reps * 100))
                           for key, val in all_nodes.items()])
df_CP_freq.sort_values(by='Freq', ascending=False, inplace=True)
df_CP_freq.set_index('Task', inplace=True)
print(df_CP_freq)

# Plot Critical Path Frequency
plt.figure(figsize=(10, 6))
df_CP_freq.plot(kind='bar', legend=False)
plt.title('Critical Path Frequency')
plt.xlabel('Task')
plt.ylabel('Frequency (%)')
plt.xticks(rotation=45)
plt.show()

# Print results table
print("\nActivity Analysis (at mean duration):")
print("Activity\tES\tEF\tLS\tLF\tFloat\tDrag\tCritical")
for activity in sorted(activities.keys()):
    data = all_nodes[activity]
    print(f"{activity}\t\t{data['ES']}\t{data['EF']}\t{data['LS']}\t{data['LF']}\t{data['float']}\t{data['drag']}\t{'Yes' if data['CP'] else 'No'}")

print("\nCritical Path:", " -> ".join([activity for activity in all_nodes if all_nodes[activity]['CP']]))

## --- Gantt Chart Rendering ---
def plot_gantt_chart(nodes):
    fig, ax = plt.subplots(figsize=(12, 8))
    yticks = []
    yticklabels = []
    for i, (activity, data) in enumerate(nodes.items()):
        ax.broken_barh([(data['ES'], data['dur'])], (i - 0.4, 0.8), facecolors=('tab:blue' if not data['CP'] else 'tab:red'))
        yticks.append(i)
        yticklabels.append(activity)
        # Add text annotations for start and finish times
        ax.text(data['ES'], i, f"ES: {data['ES']}", va='center', ha='right', fontsize=8, color='black')
        ax.text(data['EF'], i, f"EF: {data['EF']}", va='center', ha='left', fontsize=8, color='black')
        # Add text annotations for LS and LF times
        ax.text(data['LS'], i - 0.2, f"LS: {data['LS']}", va='center', ha='right', fontsize=8, color='green')
        ax.text(data['LF'], i - 0.2, f"LF: {data['LF']}", va='center', ha='left', fontsize=8, color='green')
        # Add text annotations for float and drag
        ax.text(data['EF'], i + 0.2, f"Float: {data['float']}", va='center', ha='left', fontsize=8, color='blue')
        ax.text(data['EF'], i + 0.4, f"Drag: {data['drag']}", va='center', ha='left', fontsize=8, color='red')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Activities')
    ax.set_title('Gantt Chart with Critical Path Highlighted')
    ax.grid(True)
    plt.show()

plot_gantt_chart(all_nodes)

# Export results to JSON
import json
results = {
    'project_duration_distribution': pdf.to_list(),
    'mean_project_duration': mean_project_duration,
    'percentile_95': percentile_95,
    'critical_path_frequency': df_CP_freq.to_dict(),
    'activity_analysis': {activity: all_nodes[activity] for activity in sorted(activities.keys())},
    'critical_path': [activity for activity in all_nodes if all_nodes[activity]['CP']]
}

with open('project_results.json', 'w') as f:
    json.dump(results, f, indent=4)