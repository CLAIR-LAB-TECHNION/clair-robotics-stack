"""Unified Planning (UP) utilities for PDDL-based task planning.

This module provides utilities for working with PDDL (Planning Domain Definition Language)
through the Unified Planning framework. Functions include creating UP problem instances,
manipulating states, working with predicates, and collecting planning data.
"""

from itertools import product
from typing import Optional

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import Problem, UPState, FNode, Not


def create_up_problem(domain: str, problem: str) -> Problem:
    """Create a Unified Planning Problem from PDDL domain and problem.
    Files are expected to have the .pddl extension. If the domain is a file, the problem must also be a file.

    Args:
        domain: Path to domain PDDL file or domain PDDL string
        problem: Path to problem PDDL file or problem PDDL string

    Returns:
        A Unified Planning Problem instance

    Raises:
        AssertionError: If domain is a file but problem is not
    """
    # initialize PDDL reader
    reader = PDDLReader()

    # check if domain and problem are files
    if domain.lower().endswith(".pddl"):
        # check if problem is also a file
        assert problem.lower().endswith(
            ".pddl"
        ), "if domain is a file, problem must also be a file"

        # parse domain and problem files
        up_problem = reader.parse_problem(domain, problem)
    else:
        # parse domain and problem strings
        up_problem = reader.parse_problem_string(domain, problem)

    return up_problem


def get_object_names_dict(up_problem: Problem) -> dict[str, list[str]]:
    """Get a dictionary of object names grouped by their types.

    Args:
        up_problem: Unified Planning Problem instance

    Returns:
        Dictionary mapping type names to lists of object names
    """
    objects = {}
    for t in up_problem.user_types:
        objects[t.name] = list(map(str, up_problem.objects(t)))

    return objects


def get_pddl_files_str(up_problem: Problem) -> tuple[str, str]:
    """Convert a UP problem back to PDDL domain and problem strings.
    Output may differ from the input due to the way UP represents the domain and problem.

    Args:
        up_problem: Unified Planning Problem instance

    Returns:
        Tuple of (domain_str, problem_str)
    """
    writer = PDDLWriter(up_problem)
    return writer.get_domain(), writer.get_problem()


def get_all_grounded_predicates_for_objects(
    up_problem: Problem, objects: Optional[dict[str, list[str]]] = None
) -> list[str]:
    """Generate all possible grounded predicates for given objects.

    Args:
        up_problem: Unified Planning Problem instance
        objects: Dictionary mapping type names to lists of object names
                If None, obtained from get_object_names_dict

    Returns:
        List of grounded predicate strings
    """
    predicates = up_problem.fluents
    if objects is None:
        objects = get_object_names_dict(up_problem)

    grounded_predicates = []
    for p in predicates:
        varlists = []
        for variable in p.signature:
            varlists.append(objects[variable.type.name])
        for assignment in product(*varlists):
            grounded_predicates.append(f'{p.name}({",".join(assignment)})')

    return grounded_predicates


def ground_predicate_str_to_fnode(up_problem: Problem, predicate_str: str) -> FNode:
    """Convert a string representation of a grounded predicate to a unified-planning feature.

    Args:
        up_problem: Unified Planning Problem instance
        predicate_str: String representation of grounded predicate (e.g. "at(robot,kitchen)")

    Returns:
        FNode representation of the grounded predicate
    """
    fluent_name, args = predicate_str.split("(")
    args = args.rstrip(")").split(",")
    args = [arg.strip() for arg in args if arg]
    pred_obj = up_problem.fluent(fluent_name)
    arg_obj = [up_problem.object(a) for a in args]
    if arg_obj:
        return pred_obj(*arg_obj)
    else:
        return pred_obj()


def ground_predicate_fnode_to_str(up_problem: Problem, fnode: FNode) -> str:
    """Convert a unified-planning feature to a string representation of a grounded predicate.

    Args:
        up_problem: Unified Planning Problem instance
        fnode: Unified-planning feature representing a grounded predicate
    """
    return f"{fnode.fluent().name}({','.join(map(str, fnode.args))})"


def bool_constant_to_fnode(up_problem: Problem, constant: bool) -> FNode:
    """Convert a boolean constant to a a unified-planning feature.

    Args:
        up_problem: Unified Planning Problem instance
        constant: Boolean value to convert

    Returns:
        FNode representation of the boolean constant
    """
    exp_mgr = up_problem.environment.expression_manager
    if constant is True:
        return exp_mgr.true_expression
    else:
        return exp_mgr.false_expression


def convert_state_dict_to_up_compatible(
    up_problem, state_dict: dict[str, bool]
) -> dict[FNode, FNode]:
    """Convert a state string-boolean dictionary to a unified-planning features dictionary.

    Args:
        up_problem: Unified Planning Problem instance
        state_dict: Dictionary mapping predicate strings to boolean values

    Returns:
        Dictionary mapping FNode predicates to FNode boolean values
    """
    return {
        ground_predicate_str_to_fnode(up_problem, k): bool_constant_to_fnode(
            up_problem, v
        )
        for k, v in state_dict.items()
    }


def state_dict_to_up_state(up_problem: Problem, state_dict: dict[str, bool]) -> UPState:
    """Convert a state dictionary to a UPState object.

    Args:
        up_problem: Unified Planning Problem instance
        state_dict: Dictionary mapping predicate strings to boolean values

    Returns:
        UPState object representing the state
    """
    return UPState(convert_state_dict_to_up_compatible(up_problem, state_dict))


def up_state_to_state_dict(up_state: UPState) -> dict[str, bool]:
    """Convert a UPState object to a state string-boolean dictionary.

    Args:
        up_state: UPState object

    Returns:
        Dictionary mapping predicate strings to boolean values
    """
    current_instance = up_state
    out = {}
    while current_instance is not None:
        for k, v in current_instance._values.items():
            out.setdefault(
                f'{k.fluent().name}({",".join(map(str, k.args))})', v.constant_value()
            )
        current_instance = current_instance._father

    return out


def set_problem_init_state(up_problem: Problem, init_state_dict: dict[str, bool]):
    """Set the initial state of a problem from a state string-boolean dictionary.

    Args:
        up_problem: Unified Planning Problem instance
        init_state_dict: Dictionary mapping predicate strings to boolean values
    """
    # clear existing fluents
    up_problem.explicit_initial_values.clear()

    # set desired fluents
    for k, v in convert_state_dict_to_up_compatible(
        up_problem, init_state_dict
    ).items():
        up_problem.set_initial_value(k, v)


def set_problem_goal_state(
    up_problem: Problem, goal_state_dict: dict[str, bool], include_negatives=False
):
    """Set the goal state of a problem from a state string-boolean dictionary.

    Args:
        up_problem: Unified Planning Problem instance
        goal_state_dict: Dictionary mapping predicate strings to boolean values
        include_negatives: Whether to include negative goals (False predicates)
    """
    # clear existing goals
    up_problem.clear_goals()

    # set desired goals
    for k, v in goal_state_dict.items():
        if v is True:
            up_problem.add_goal(
                ground_predicate_str_to_fnode(up_problem, k),
            )
        elif include_negatives:
            up_problem.add_goal(
                Not(ground_predicate_str_to_fnode(up_problem, k)),
            )
