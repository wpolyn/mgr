import pickle

#load registry from .pkl
def load_registry():
    print("Loading the registry...")
    label_to_id = {}
    try:
        with open('pickles/label_to_id.pkl', 'rb') as f:
            label_to_id = pickle.load(f)
        print(f"Registry loaded successfully:\n{label_to_id}")
    except EOFError:
        print("Registry file is empty or corrupted.")
    except FileNotFoundError:
        print("Registry file not found.")
    return label_to_id

#invert registry for identification by ID
def invert_registry(label_to_id):
    id_to_label = {ID: label for label, ID in label_to_id.items()}
    return id_to_label

#save registry from label_to_id
def save_registry(label_to_id):
    print("Saving the registry...")
    registry = 'pickles/label_to_id.pkl'
    with open(registry, 'wb') as f:
        pickle.dump(label_to_id, f)
    print("Registry saved successfully.") 