

def calculate_num_classes(annotations):
    unique_targets = set()
    for anno in annotations:
        unique_targets = unique_targets.union(anno['labelId'])
    print(sorted(list(unique_targets)))
    num_classes = max(list(map(int, unique_targets)))
    # add one since zero is a class
    return num_classes + 1