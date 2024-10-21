import inspect
from datetime import datetime
from typing import get_type_hints, Any


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func: callable) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # Parse docstring
    docstring = inspect.getdoc(func) or ""
    description, arg_descriptions = parse_docstring(docstring)

    # Get type hints
    type_hints = get_type_hints(func)

    parameters = {}
    required = []

    for param_name, param in signature.parameters.items():
        try:
            param_type = type_hints.get(param_name, Any)
            param_type_str = type_map.get(param_type, param_type.__name__)
            if param_type_str == "Any":
                param_type_str = "string"  # Default to string if no type hint
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param_name}: {str(e)}"
            )

        param_info = {"type": param_type_str}

        # Only add description if Args section is present and this param is described
        if arg_descriptions and param_name in arg_descriptions:
            param_info["description"] = arg_descriptions[param_name]

        parameters[param_name] = param_info

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def parse_docstring(docstring: str) -> tuple[str, dict]:
    """
    Parse a function's docstring to extract the main description and argument descriptions.

    Args:
        docstring (str): The function's docstring.

    Returns:
        tuple: A tuple containing the main description and a dictionary of argument descriptions.
    """
    description = []
    arg_descriptions = {}
    current_param = None
    in_args_section = False

    lines = docstring.split("\n")

    for line in lines:
        line = line.strip()

        # Check for the start of the Args section
        if line.lower().startswith("args:"):
            in_args_section = True
            continue

        if not in_args_section:
            # If we're not in the Args section, add to the main description
            description.append(line)
        else:
            # We're in the Args section
            if ":" in line:
                # This line defines a new parameter
                parts = line.split(":", 1)
                current_param = parts[0].strip()
                param_desc = parts[1].strip()
                arg_descriptions[current_param] = param_desc
            elif current_param and line:
                # This line continues the description of the current parameter
                arg_descriptions[current_param] += " " + line

    # Join the description lines
    full_description = " ".join(description).strip()

    # If no Args section was found, return an empty dict for arg_descriptions
    return full_description, arg_descriptions if in_args_section else {}
