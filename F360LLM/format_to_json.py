import json
import re


def format_to_jsonable(input_string):
   
    try:
        structured_data = {}
        parameters = {}

        # Extract each component using regular expressions
        shape_match = re.search(r'shape:\s*([^,]+)', input_string)
        if shape_match:
            structured_data['shape'] = shape_match.group(1).strip()

        parameters_match = re.search(r'parameters:\s*([^,]+(?:,\s*\w+:\s*[\w\.]+)*)', input_string)
        if parameters_match:
            param_string = parameters_match.group(1).strip()
            param_pairs = param_string.split(", ")
            for param in param_pairs:
                param_key, param_value = param.split(":")
                param_key = param_key.strip()
                param_value = param_value.strip()

            
                try:
                    param_value = int(param_value)
                except ValueError:
                    pass

                parameters[param_key] = param_value

 
        structured_data["parameters"] = parameters

        plane_match = re.search(r'plane:\s*([^,]+)', input_string)
        if plane_match:
            plane_value = plane_match.group(1).strip()
            structured_data['plane'] = plane_value

        
            if "plane" in parameters:
                del parameters["plane"]

        coordinates_match = re.search(r'coordinates:\s*(\[[^\]]+\])', input_string)
        if coordinates_match:
            coordinates_str = coordinates_match.group(1).strip()
            coordinates = []
            if coordinates_str.startswith("[") and coordinates_str.endswith("]"):
                coordinates_str = coordinates_str[1:-1]  # Remove brackets
                coord_parts = coordinates_str.split(",")
                for coord in coord_parts:
                    coord = coord.strip()
                    try:
                        coordinates.append(int(coord))
                    except ValueError:
                        print(f"Warning: Could not convert coordinate '{coord}' to an integer")

            structured_data["coordinates"] = coordinates

        return structured_data

    except Exception as e:
        print(f"Error in formatting: {e}")
        return None

def test_format_to_jsonable():


    for input_string in inputs:
        print(f"Input: {input_string}")
        formatted_output = format_to_jsonable(input_string)
        if formatted_output:
            print(f"Formatted JSON:\n{json.dumps(formatted_output, indent=4)}\n")
        else:
            print("Failed to format the input.\n")
