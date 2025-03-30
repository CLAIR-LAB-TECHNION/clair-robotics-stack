from abc import ABC, abstractmethod
import json
import os
from typing import Callable

from timeoutcontext import timeout
from tqdm.auto import tqdm
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.shortcuts import Problem, OneshotPlanner, PlanValidator

from .up_utils import *

from .action_executer import ActionExecuter
from .state_estimator import ThreeLayerStateEstimator


class TAMPRunnerCallbacks(ABC):
    """
    Abstract base class for callbacks to TAMPRunner.
    """

    def __init__(self, runner: "TAMPRunner", *args, **kwargs):
        self.runner = runner

    @abstractmethod
    def on_episode_start(self):
        pass

    @abstractmethod
    def on_replan(self):
        pass

    @abstractmethod
    def on_action_start(self, action):
        pass

    @abstractmethod
    def on_action_end(self, action, success):
        pass

    @abstractmethod
    def on_episode_end(self):
        pass


class TAMPRunner:
    def __init__(
        self,
        problem: Problem,
        executer: ActionExecuter,
        state_estimator: ThreeLayerStateEstimator,
        sensor_fn: Callable[[], dict],
        max_actions: int = 100,
        max_action_retries: int = 3,
        action_timeout: int = 60,
        callbacks: TAMPRunnerCallbacks = [],
    ):
        self.problem = problem
        self.executer = executer
        self.state_estimator = state_estimator
        self.sensor_fn = sensor_fn
        self.max_actions = max_actions
        self.max_action_retries = max_action_retries
        self.action_timeout = action_timeout
        self.callbacks = callbacks

        # initialize states
        self.cur_task_state = None
        self.cur_motion_state = None

        # initialize planning components
        # TODO configurable planner
        self.problem_sim = UPSequentialSimulator(problem)
        self.planner = OneshotPlanner(name="fast-downward")

        self.cur_plan = None

    def run_episode(self):
        ################
        # Episode Init #
        ################

        self.callbacks.on_episode_start()

        # perceive initial state
        self.update_states()

        # init counters
        action_count = 0
        failures_count = 0

        # check if initial state is a goal state
        if self.problem_sim.is_goal(self.cur_task_state):
            print("initial predicted state is goal state. skipping episode")
            self.callbacks.on_episode_end()
            return

        ###############
        # Episode Run #
        ###############

        while (
            action_count < self.max_episode_actions
            or failures_count < self.max_action_failures
        ):

            # get next action
            action = self.get_next_action()
            if action is None:
                print("no plan from current predicted state. resetting")
                break

            # apply action
            suc = self.apply_action(action)
            if not suc:
                failures_count += 1
                continue

            # update task state
            self.update_states()

            if self.problem_sim.is_goal(self.cur_task_state):
                print("predicting goal reached")
                break

            action_count += 1

        self.callbacks.on_episode_end()

    def update_states(self):
        observations = self.sensor_fn()
        self.cur_task_state, self.cur_motion_state, _ = (
            self.state_estimator.estimate_state(observations)
        )

    def get_next_action(self):
        # set problem initial state
        state_dict = up_state_to_state_dict(self.cur_task_state)
        set_problem_init_state(self.problem, state_dict)

        # check if current plan is still valid
        if self.cur_plan is not None and len(self.cur_plan.actions) > 0:
            # validate plan
            with PlanValidator(
                problem_kind=self.problem.kind, plan_kind=self.cur_plan.kind
            ) as validator:

                if validator.validate(self.problem, self.cur_plan):
                    # plan still valid. get next action
                    return self.cur_plan._actions.pop(0)

        # get new plan
        plan_res = self.planner.solve(self.problem)
        if plan_res.status != PlanGenerationResultStatus.SOLVED_SATISFICING:
            return None
        self.cur_plan = plan_res.plan

        self.callbacks.on_replan()

        return self.cur_plan._actions.pop(0)

    def apply_action(self, action):
        self.callbacks.on_action_start(action)

        # extract action name and params
        action_name, params = action.action.name, action.actual_parameters
        params = [str(param) for param in params]

        # run action
        for _ in range(self.max_action_retries):
            # attempt action
            try:
                with timeout(self.action_timeout):  # one minute timeout
                    suc = self.executer.execute_action(
                        action_name, params, self.cur_motion_state
                    )
            except Exception as e:
                print(
                    f"failed to execute action {action_name}({','.join(params)}) with error: {e}"
                )
                continue  # failure. continue loop

            if not suc:
                print(f"action {action_name}({','.join(params)}) resulted in failure")
                continue  # failure. continue loop

        self.callbacks.on_action_end(action, suc)
        return suc

    def __del__(self):
        self.planner.destroy()
