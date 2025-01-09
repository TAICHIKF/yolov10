# Define the scales dictionary
scales = {
    "n": [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    "s": [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    "m": [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    "l": [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    "x": [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
}

# Define the extraction function
def extract_parameters(scale_key):
    results = {
        "layers": None,
        "parameters": None,
        "gradients": None,
        "GFLOPs": None,
    }
    
    # Use conditional statements to extract parameters based on the key
    if scale_key == "n":
        results.update({
            "layers": 225,
            "parameters": 3000000,
            "gradients":  3157184,
            "GFLOPs": 8.9
        })
    elif scale_key == "s":
        results.update({
            "layers": 225,
            "parameters": 10000000,
            "gradients":  11166544,
            "GFLOPs": 28.8
        })
    elif scale_key == "m":
        results.update({
            "layers": 295,
            "parameters": 25000000,
            "gradients":  25902624,
            "GFLOPs": 79.3
        })
    elif scale_key == "l":
        results.update({
            "layers": 365,
            "parameters": 40000000,
            "gradients":  43691504,
            "GFLOPs": 165.7
        })
    elif scale_key == "x":
        results.update({
            "layers": 365,
            "parameters": 60000000,
            "gradients":  68229632,
            "GFLOPs": 258.5
        })
    else:
        raise ValueError(f"Unknown scale key: {scale_key}")

    return results


        
if __name__ == '__main__':

    # Ask for a specific key and process it
    key = input("Enter the scale key (n, s, m, l, x): ").strip()

    if key in scales:
        value = scales[key]
        params = extract_parameters(key)
        print(params)
        
        # print(f"Scale '{key}' details:")
        # print(f"  Depth: {value[0]}, Width: {value[1]}, Max Channels: {value[2]}")
        # print(f"  Layers: {params['layers']}, Parameters: {params['parameters']}, Gradients: {params['gradients']}, GFLOPs: {params['GFLOPs']}")
    else:
        print(f"Invalid key '{key}'. Please enter one of: {', '.join(scales.keys())}")