import pickle

# Load the .pkl file
file_path = '/media/liyuqiu/RS-PCT/data/xView/ss/trainval/annfiles/json/patch_annfile.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Determine the type of the loaded data and its structure (if applicable)
data_type = type(data)
if isinstance(data, (list, dict, set, tuple)):
    structure = {type(data)}
    if isinstance(data, dict):
        key_types = {type(k) for k in data.keys()}
        value_types = {type(v) for v in data.values()}
        structure.update({"keys": key_types, "values": value_types})
    elif isinstance(data, (list, set, tuple)):
        element_types = {type(element) for element in data}
        structure.update({"elements": element_types})
else:
    structure = "N/A"  # For non-iterable or simple data types

print(data_type, structure)
