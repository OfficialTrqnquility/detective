def flattenArrayTransform(data):
    # Initialize new flattened array
    mapped = []

    # Loop through the items in the data dictionary
    for value in data:
        # If is a regular value append the map if it is a dict do recursion.
        if isinstance(value, list):
            mapped.extend(flattenArrayTransform(value))
        # Otherwise, add the value to the map.
        else:
            mapped.append(value)

    # Return the mapping array
    return mapped