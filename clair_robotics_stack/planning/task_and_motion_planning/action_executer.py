from abc import ABC
from typing import Any
import inspect


class ActionExecuter(ABC):
    """
    Abstract base class for executing actions in a task and motion planning system.
    
    This class defines the interface for executing actions from a planning domain.
    Concrete implementations should:
    
    1. Inherit from this class
    2. Implement methods corresponding to each action in the planning domain
    3. Each action method should:
       - Accept parameters matching those defined in the planning domain
       - Return a boolean indicating success or failure of the action
       - Handle any necessary robot control or environment interaction
    
    For example, if the planning domain has an action "pick-object", the implementation
    should have a method named "pick_object" (note the underscore replacing the hyphen)
    that handles the actual execution of picking up an object with the robot.
    
    The execute_action method automatically converts action names from PDDL format
    (with hyphens) to Python method names (with underscores) and dispatches to the
    appropriate method.
    """
    def __init__(self) -> None:
        self._motion_state = None

    def execute_action(self, action: str, parameters: list[Any], motion_state) -> bool:
        """
        Execute an action with the given parameters.

        Args:
            action: The name of the action to execute.
            parameters: The parameters of the action.

        Returns:
            True if the action was executed successfully, False otherwise.
        """
        # replace PDDL legal hyphens to python legal underscores
        action = action.replace("-", "_")

        # get skill function from class attributes
        try:
            skill = getattr(self, action)
        except AttributeError:
            raise AttributeError(
                f"action mapper {self.__class__.__name__} does not support action '{action}'"
            )
        
        self._motion_state = motion_state

        # execute skill and return success
        return skill(*parameters)


def check_actions_compatibility(problem, action_executer):
    """
    Check if the actions of a unified-planning problem are compatible with a given action executer.
    
    Args:
        problem: A unified-planning Problem instance.
        action_executer: An instance of ActionExecuter or its subclass.
    
    Returns:
        tuple: (is_compatible, incompatible_actions) where:
            - is_compatible (bool): True if all actions in the problem are supported by the executer
                with matching parameter counts
            - incompatible_actions (dict): Dictionary mapping action names to reasons they're incompatible
                (either "missing" or "parameter_mismatch")
    """
    incompatible_actions = {}
    
    for action in problem.actions:
        action_name = action.name.replace("-", "_")
        
        # Check if the action method exists
        if not hasattr(action_executer, action_name):
            incompatible_actions[action.name] = "missing"
            continue
            
        # Check if parameter counts match
        method = getattr(action_executer, action_name)
        expected_param_count = len(action.parameters)
        actual_param_count = len(inspect.signature(method).parameters)
        
        if expected_param_count != actual_param_count:
            incompatible_actions[action.name] = f"parameter_mismatch (expects {expected_param_count}, found {actual_param_count})"
    
    return len(incompatible_actions) == 0, incompatible_actions

