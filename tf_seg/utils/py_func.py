def pascal_case_to_snake_case(s: str) -> str:
    """Convert class name to snake case name."""
    return "".join(["_" + i.lower() if i.isupper() else i for i in s]).lstrip("_")


def snake_case_to_pascal_case(s):
    """Convert snake case name to pascal name."""
    return "".join(i.capitalize() for i in s.split("_"))
