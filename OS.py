import random
import matplotlib.pyplot as plt

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
C = 1e-9  # Switching capacitance
FREQ_LEVELS = [0.8e9, 1.2e9, 1.6e9, 2.0e9]  # in Hz
VOLTAGE_LEVELS = [0.8, 0.9, 1.0, 1.1]
TIME_SLICE = 4
LOW_UTIL_THRESHOLD = 0.4
HIGH_UTIL_THRESHOLD = 0.8

# -----------------------------------------------
# TASK GENERATION
# -----------------------------------------------
def generate_tasks(n=10):
    tasks = []
    for i in range(n):
        tasks.append({
            "id": i + 1,
            "burst": random.randint(3, 12),
            "arrival": random.randint(0, 5 * i)
        })
    return sorted(tasks, key=lambda x: x["arrival"])

# -----------------------------------------------
# POWER & ENERGY MODEL
# -----------------------------------------------
def compute_power(freq, voltage):
    return C * (voltage ** 2) * freq

def compute_energy(power, time):
    return power * time

# -----------------------------------------------
# DVFS CONTROLLER
# -----------------------------------------------
def dvfs_control(utilization):
    """Return freq and voltage based on CPU utilization"""
    if utilization < LOW_UTIL_THRESHOLD:
        return FREQ_LEVELS[0], VOLTAGE_LEVELS[0]
    elif utilization < HIGH_UTIL_THRESHOLD:
        return FREQ_LEVELS[1], VOLTAGE_LEVELS[1]
    else:
        return FREQ_LEVELS[-1], VOLTAGE_LEVELS[-1]

# -----------------------------------------------
# COMMON METRIC CALCULATIONS
# -----------------------------------------------
def calculate_metrics(tasks, completion_times):
    turnaround_times = []
    waiting_times = []
    for task, comp in zip(tasks, completion_times):
        tat = comp - task["arrival"]
        wt = tat - task["burst"]
        turnaround_times.append(tat)
        waiting_times.append(wt)
    avg_tat = sum(turnaround_times) / len(tasks)
    avg_wt = sum(waiting_times) / len(tasks)
    total_time = max(completion_times)
    return avg_tat, avg_wt, total_time

# -----------------------------------------------
# ENERGY-EFFICIENT SCHEDULER (DVFS BASED)
# -----------------------------------------------
def energy_efficient_scheduler(tasks):
    time = 0
    total_energy = 0
    ready_queue = []
    remaining = tasks.copy()
    freq_trace, volt_trace, time_trace = [], [], []
    completion_times = []

    while remaining or ready_queue:
        for task in remaining[:]:
            if task["arrival"] <= time:
                ready_queue.append(task)
                remaining.remove(task)
        if not ready_queue:
            time += 1
            freq_trace.append(FREQ_LEVELS[0])
            volt_trace.append(VOLTAGE_LEVELS[0])
            time_trace.append(time)
            continue

        current = min(ready_queue, key=lambda x: x["burst"])
        ready_queue.remove(current)
        utilization = min(1.0, current["burst"] / 15)
        freq, volt = dvfs_control(utilization)
        power = compute_power(freq, volt)
        energy = compute_energy(power, current["burst"])
        total_energy += energy
        time += current["burst"]

        # Track dynamic DVFS behavior
        freq_trace.append(freq)
        volt_trace.append(volt)
        time_trace.append(time)
        completion_times.append(time)

    avg_tat, avg_wt, total_time = calculate_metrics(tasks, completion_times)
    return total_energy, avg_tat, avg_wt, total_time, freq_trace, volt_trace, time_trace

# -----------------------------------------------
# BASELINE ALGORITHMS
# -----------------------------------------------
def fcfs_scheduler(tasks):
    time, total_energy = 0, 0
    completion_times = []
    for task in tasks:
        if time < task["arrival"]:
            time = task["arrival"]
        freq, volt = FREQ_LEVELS[-1], VOLTAGE_LEVELS[-1]
        power = compute_power(freq, volt)
        energy = compute_energy(power, task["burst"])
        total_energy += energy
        time += task["burst"]
        completion_times.append(time)
    return total_energy, *calculate_metrics(tasks, completion_times)

def sjf_scheduler(tasks):
    time, total_energy = 0, 0
    ready_queue, remaining = [], tasks.copy()
    completion_times = []
    while remaining or ready_queue:
        for task in remaining[:]:
            if task["arrival"] <= time:
                ready_queue.append(task)
                remaining.remove(task)
        if not ready_queue:
            time += 1
            continue
        current = min(ready_queue, key=lambda x: x["burst"])
        ready_queue.remove(current)
        freq, volt = FREQ_LEVELS[-1], VOLTAGE_LEVELS[-1]
        power = compute_power(freq, volt)
        energy = compute_energy(power, current["burst"])
        total_energy += energy
        time += current["burst"]
        completion_times.append(time)
    return total_energy, *calculate_metrics(tasks, completion_times)

def rr_scheduler(tasks):
    time, total_energy = 0, 0
    queue = []
    remaining = {t["id"]: t["burst"] for t in tasks}
    tasks_copy = tasks.copy()
    completion_times = {}
    while remaining:
        for task in tasks_copy[:]:
            if task["arrival"] <= time:
                queue.append(task)
                tasks_copy.remove(task)
        if not queue:
            time += 1
            continue
        current = queue.pop(0)
        burst_time = min(TIME_SLICE, remaining[current["id"]])
        freq, volt = FREQ_LEVELS[-1], VOLTAGE_LEVELS[-1]
        power = compute_power(freq, volt)
        energy = compute_energy(power, burst_time)
        total_energy += energy
        remaining[current["id"]] -= burst_time
        time += burst_time
        if remaining[current["id"]] > 0:
            queue.append(current)
        else:
            completion_times[current["id"]] = time
            del remaining[current["id"]]
    completion_order = [completion_times[t["id"]] for t in tasks]
    return total_energy, *calculate_metrics(tasks, completion_order)

# -----------------------------------------------
# RUN SIMULATION
# -----------------------------------------------
tasks = generate_tasks(10)

algorithms = {
    "FCFS": fcfs_scheduler,
    "SJF": sjf_scheduler,
    "RR": rr_scheduler,
    "Energy Efficient (DVFS)": energy_efficient_scheduler
}

results = {}
for name, algo in algorithms.items():
    print(f"\nRunning {name} Scheduler...")
    res = algo(tasks)
    results[name] = res

# -----------------------------------------------
# RESULTS SUMMARY
# -----------------------------------------------
print("\n--- Energy & Performance Summary ---")
print("Algorithm\t\tEnergy(J)\tAvg TAT\tAvg WT\tTotal Time")
for k, v in results.items():
    energy, tat, wt, total_time = v[:4]
    print(f"{k:22s}{energy:.4e}\t{tat:.2f}\t{wt:.2f}\t{total_time:.2f}")

# -----------------------------------------------
# VISUALIZATION
# -----------------------------------------------
energies = [results[k][0] for k in results.keys()]
turnaround = [results[k][1] for k in results.keys()]

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(results.keys(), energies)
plt.title("Energy Consumption Comparison")
plt.ylabel("Total Energy (J)")
plt.xticks(rotation=15)

plt.subplot(1,2,2)
plt.bar(results.keys(), turnaround, color='orange')
plt.title("Average Turnaround Time Comparison")
plt.ylabel("Avg Turnaround Time (s)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# DVFS Behavior Plot
# -----------------------------------------------
freq_trace = results["Energy Efficient (DVFS)"][4]
volt_trace = results["Energy Efficient (DVFS)"][5]
time_trace = results["Energy Efficient (DVFS)"][6]

plt.figure(figsize=(10,5))
plt.plot(time_trace, [f/1e9 for f in freq_trace], label='Frequency (GHz)')
plt.plot(time_trace, volt_trace, label='Voltage (V)')
plt.title("DVFS Frequency & Voltage Scaling Over Time")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.show()
