def removeDictKeyTransform(data):
    # Initialize the mapping array
    mapped = []

    # Loop through the items in the data dictionary
    for key, value in data.items():
        # If the value is a dictionary, convert its values to a numpy array and add it to the mapping array
        if isinstance(value, dict):
            mapped.append(list(value.values()))
        # Otherwise, convert the value to a numpy array and add it to the mapping array
        else:
            mapped.append(value)

    # Return the mapping array
    return mapped
