import pickle

#memory utilization history handling
def _load_mem_history():
    try:
        with open('mem_history.pkl', 'rb') as f:
            mem_history = pickle.load(f)
            return mem_history
    except FileNotFoundError:
        print("Memory utilization history file not found. Creating a new file.")
        return {}

def _save_mem_history(mem_history):
    with open('mem_history.pkl', 'wb') as f:
        pickle.dump(mem_history, f)

def append_mem_history(model_name, mem_avg):
    history = _load_mem_history()
    if model_name not in history:
        history[model_name] = []
    history[model_name].append(mem_avg)
    print(f"Updated memory history for {model_name}: {history[model_name]}")
    _save_mem_history(history)

#cpu utilization history handling
def _load_cpu_history():
    try:
        with open('cpu_history.pkl', 'rb') as f:
            cpu_history = pickle.load(f)
            return cpu_history
    except FileNotFoundError:
        print("CPU utilization history file not found. Creating a new file.")
        return {}

def _save_cpu_history(cpu_history):
    with open('cpu_history.pkl', 'wb') as f:
        pickle.dump(cpu_history, f)

def append_cpu_history(model_name, cpu_avg):
    history = _load_cpu_history()
    if model_name not in history:
        history[model_name] = []
    history[model_name].append(cpu_avg)
    print(f"Updated CPU history for {model_name}: {history[model_name]}")
    _save_cpu_history(history)

#duration history handling
def _load_duration_history():
    try:
        with open('duration_history.pkl', 'rb') as f:
            duration_history = pickle.load(f)
            return duration_history
    except FileNotFoundError:
        print("Duration history file not found. Creating a new file.")
        return {}
    
def _save_duration_history(duration_history):
    with open('duration_history.pkl', 'wb') as f:
        pickle.dump(duration_history, f)

def append_duration_history(model_name, duration):
    history = _load_duration_history()
    if model_name not in history:
        history[model_name] = []
    history[model_name].append(duration)
    print(f"Updated duration history for {model_name}: {history[model_name]}")
    _save_duration_history(history)