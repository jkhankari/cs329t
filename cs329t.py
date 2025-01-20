from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any, Tuple
import openai
from enum import Enum
from queue import PriorityQueue
import json
import copy
import itertools

from dataclasses import dataclass

#TODO: seperate out model and temperature into separate parameters to pass in but with defaults

class ControlType(Enum):
    SEQUENTIAL = "sequential"
    LOOP = "loop"
    CONDITIONAL = "conditional"

@dataclass
class ReasoningStrategy:
    name: str
    description: str
    prompt_template: str

    @classmethod
    def get_strategies(cls) -> Dict[str, 'ReasoningStrategy']: #returns a dictionary with reasoning stragetgies
        return {
            "sanity_check": cls(
                "sanity_check",
                "Verify if the result makes logical sense",
                "Examine this result carefully:\n{result}\n"
                "1. Does it make numerical sense?\n"
                "2. Are the units consistent?\n"
                "3. Is it within reasonable bounds?\n"
            ),
            "counter_example": cls(
                "counter_example",
                "Try to find cases where this might fail",
                "Consider potential counter-examples:\n{result}\n"
                "1. What edge cases might break this?\n"
                "2. Are there boundary conditions to consider?\n"
                "3. What assumptions are we making?\n"
            ),
            "logical_proof": cls(
                "logical_proof",
                "Prove the logic step by step",
                "Let's prove this logically:\n{result}\n"
                "1. Start with our givens\n"
                "2. Apply each logical step\n"
                "3. Verify each inference\n"
            ),
            "global_constraints": cls(
                "global_constraints",
                "Check against problem constraints",
                "Verify against our constraints:\n{result}\n"
                "1. Does this satisfy all given conditions?\n"
                "2. Does it maintain required invariants?\n"
                "3. Does it conflict with any problem rules?\n"
            ),
            "creative_analogy": cls(
                "creative_analogy",
                "Use analogical reasoning",
                "Think of this through analogy:\n{result}\n"
                "1. What similar problems have we solved?\n"
                "2. How does this map to familiar concepts?\n"
                "3. Can we verify through parallel reasoning?\n"
            )
        }

@dataclass
class PlanStep:
    id: int
    description: str
    control_type: ControlType
    input_spec: Dict
    output_spec: Dict
    reasoning_strategies: List[str]
    condition: Optional[str] = None  # For loops and conditionals
    sub_steps: List['PlanStep'] = None
    max_iterations: Optional[int] = None  # For loops

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'description': self.description,
            'control_type': self.control_type.value,
            'input_spec': self.input_spec,
            'output_spec': self.output_spec
            #'reasoning_strategies': self.reasoning_strategies,
            #'condition': self.condition,
            #'max_iterations': self.max_iterations
        }

class ExecutionState:
    def __init__(self):
        self.valid_states: List[Dict] = []  # Stack of valid execution states
        self.explored_paths: Set[str] = set()
        self.current_context: Dict = {}
        self.priority_queue = PriorityQueue()  # For managing backtracking
        self.counter = itertools.count()


    #TODO: make this search better. For instance, the hashing doesn't make sense since each state is not the same as the other even if the thought patterns are similar
    def push_state(self, state: Dict, score: float):
        state_hash = json.dumps(state, sort_keys=True)
        if state_hash not in self.explored_paths:
            self.explored_paths.add(state_hash)
            self.valid_states.append(state)
            self.priority_queue.put((-score, next(self.counter), state))  # Negative score for max-heap

    def pop_best_state(self) -> Optional[Dict]:
        if not self.priority_queue.empty():
            return self.priority_queue.get()[2]
        return None



class StrategicPlanner:
    """Responsible for high-level planning without implementation details."""

    def __init__(self, api_key: str, use_model = 'gpt-4o-mini'):
        self.api_key = api_key
        self.model = use_model

    def get_completion(self, prompt: str, json = False, use_model = "gpt-4o-mini") -> str:
        if json:
            response = openai.ChatCompletion.create(model=use_model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        else:
            response = openai.ChatCompletion.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
        return response.choices[0].message.content

    def create_strategic_plan(self, problem: str) -> List[Dict]:
        """Create a high-level plan focusing on strategy and approach."""
        prompt = f"""Analyze the following problem to determine the most suitable approach for solving it from the list of possible approaches provided.

Possible approaches:
- Divide and conquer: Split a big problem into smaller parts and solve each part separately.
- Voting: Combine answers from multiple models and pick the most popular one.
- Search: Look through different options to find the best solution.
- Debate: Have models argue different sides to figure out the best answer.
- Planning: Make a step-by-step plan to solve the problem.

Problem: {problem}

Your tasks are:
1. Parse the problem to fully understand its requirements and challenges.
2. Choose the most appropriate approach from the list above. WRITE OUT THE APPROACH.
3. Based on the chosen approach, create a high-level strategic plan that assigns helper-agents to specific tasks necessary for solving the problem.

For each helper-agent assignment, provide:
1. A clear description of the task assigned to the helper-agent.
2. Why this task is necessary within the chosen approach.
3. What information or resources the helper-agent needs.
4. What the helper-agent should produce.
5. Any dependencies on other tasks or helper-agents.

Format each assignment as:
Helper-Agent N:
Task: [specific task]
Purpose: [what this task accomplishes in the context of the approach you chose]
Rationale: [why this task is needed in the context of the approach you chose]
Needs: [information/resources needed]
Produces: [expected output]
Dependencies: [any tasks or agents this depends on]
"""


        response = self.get_completion(prompt, use_model = self.model)

        response = self._parse_strategic_plan(response)
        return response

    def _parse_strategic_plan(self, response: str) -> List[Dict]:
        """Parse the strategic plan into structured format."""
        steps = []
        current_step = {}

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                current_step = {'id': len(steps) + 1}
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                current_step[key] = value

        if current_step:
            steps.append(current_step)

        return steps

class TacticalPlanner:
    """Converts strategic plans into detailed executable steps."""

    def __init__(self, api_key: str, use_model = 'gpt-4o'):
        self.api_key = api_key
        self.model = use_model

    def get_completion(self, prompt: str, json = False, use_model = "gpt-4o") -> str:
        if json:
            response = openai.ChatCompletion.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
        else:
            response = openai.ChatCompletion.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
        return response.choices[0].message.content

    def create_tactical_plan(self, strategic_plan: List[Dict], problem: str) -> List[PlanStep]:
        """Convert strategic plan into detailed tactical steps."""
        prompt = f"""Convert this strategic plan into detailed tactical steps with specific implementation details.

Problem: {problem}

Strategic Plan:
{json.dumps(strategic_plan, indent=2)}

For each strategic step, create one or more tactical steps with:
1. Specific control type (sequential/loop/conditional)
2. Detailed input/output specifications
3. Required reasoning strategies

Return as JSON with this structure:
{{
    "steps": [
        {{
            "id": number,
            "description": string,
            "control_type": "sequential"|"loop"|"conditional",
            "input_spec": object,
            "output_spec": object,
            "reasoning_strategies": string[]
        }}
    ]
}}
"""

        response = self.get_completion(prompt, json=True, use_model =self.model)

        try:
            tactical_plan = json.loads(response)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Response content: {response}")
            tactical_plan = strategic_plan

        return [self._create_plan_step(step_dict)
                for step_dict in tactical_plan['steps']]

    def _create_plan_step(self, step_dict: Dict) -> PlanStep:
        """Create a PlanStep object from dictionary."""
        sub_steps = None
        if 'sub_steps' in step_dict and step_dict['sub_steps']:
            sub_steps = [self._create_plan_step(sub) for sub in step_dict['sub_steps']]

        #TODO: Fix the control logic here
        return PlanStep(
            id=step_dict['id'],
            description=step_dict['description'],
            #control_type=ControlType(step_dict['control_type']),
            control_type = "sequential",
            input_spec=step_dict['input_spec'],
            output_spec=step_dict['output_spec'],
            reasoning_strategies=step_dict['reasoning_strategies']
            #condition=step_dict.get('condition'),
            #sub_steps=sub_steps,
            #max_iterations=step_dict.get('max_iterations')
        )

class EnhancedHierarchicalPlanner:
    """Main planner that coordinates strategic and tactical planning with execution."""

    def __init__(self, api_key: str, models = 'gpt-4o-mini'):
        if models == 'gpt-4o-mini':
            self.model = models
        else:
            self.model = models['EnhancedHierarchicalPlanner']
            strategic_model = models['StrategicPlanner']
            tactical_model = models['TacticalPlanner']
            self.integrate_model = models['Integrate']
            self.step_models = models['Steps']
        self.api_key = api_key
        self.strategic_planner = StrategicPlanner(api_key, use_model = strategic_model)
        self.tactical_planner = TacticalPlanner(api_key, use_model = tactical_model)
        self.execution_state = ExecutionState()
        self.reasoning_strategies = ReasoningStrategy.get_strategies()




    def get_completion(self, prompt: str, json = False, model = "gpt-4o-mini", step_counter = None) -> str:
        """Helper function to get LLM completion."""
        if step_counter is not None:
            try:
                use_model = self.step_models[step_counter]
            except IndexError:
                use_model = self.model
        else:
            use_model = model
        if json:
            response = openai.ChatCompletion.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
        else:
            response = openai.ChatCompletion.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
        return response.choices[0].message.content

    def solve(self, problem: str) -> Dict:
        """Main solving function with planning and execution."""
        strategic_plan = self.strategic_planner.create_strategic_plan(problem)

        tactical_steps = self.tactical_planner.create_tactical_plan(
            strategic_plan, problem)

        return self._execute_plan(tactical_steps, problem)

    def _execute_plan(self, steps: List[PlanStep], problem: str) -> Dict:
        """Execute a tactical plan with support for replanning."""
        context = {'original_problem': problem}
        results = []
        replan_attempts = 0
        max_replan_attempts = 3

        while replan_attempts < max_replan_attempts:
            success, results, failed_step = self._try_execute_plan(
                steps, context, results)

            if success:
                final_answer = self.integrate_outputs(results, model = self.integrate_model)
                return {
                    'success': True,
                    'steps': steps,
                    'intermediate_results': results,
                    'final_answer': final_answer
                }

            new_plan = self._replan(problem, context, failed_step, results)
            if new_plan is None:
                break

            steps = new_plan
            replan_attempts += 1

        return {
            'success': False,
            'error': 'Max replan attempts reached',
            'partial_results': results
        }

    def _try_execute_plan(
        self,
        steps: List[PlanStep],
        context: Dict,
        previous_results: List
    ) -> Tuple[bool, List, Optional[PlanStep]]:
        """Attempt to execute a plan, returning success status and results."""
        results = copy.deepcopy(previous_results)

        step_counter = 0

        for step in steps:
            result = self.execute_step(step, context, step_counter=step_counter)

            if not result['success']:
                return False, results, step

            results.append(result)
            context[f'step_{step.id}_output'] = result
            step_counter += 1

        return True, results, None

    def _replan(
        self,
        problem: str,
        context: Dict,
        failed_step: PlanStep,
        partial_results: List
    ) -> Optional[List[PlanStep]]:
        """Create a new plan based on execution failure."""
        prompt = f"""A plan has failed at this step:
        Failed Step: {failed_step.description}

        Context:
        Original Problem: {problem}
        Partial Results: {json.dumps(partial_results, indent=2)}
        Current State: {json.dumps(context, indent=2)}

        Create a new plan that:
        1. Takes into account what we've learned from the failure
        2. Uses different approaches for the failed step
        3. Maintains consistency with successful steps
        4. Provides alternative paths to the solution

        Return a complete strategic plan focusing on the remaining work.
        """

        try:
            new_strategic_plan = self.strategic_planner.create_strategic_plan(prompt)

            return self.tactical_planner.create_tactical_plan(
                new_strategic_plan, problem)
        except Exception as e:
            print(f"Replanning failed: {e}")
            return None

    def execute_step(self, step: PlanStep, context: Dict, step_counter = None) -> Dict:
        """Execute a single step with validation and potential backtracking."""
        if step.control_type == ControlType.LOOP:
            return self._execute_loop(step, context)
        elif step.control_type == ControlType.CONDITIONAL:
            return self._execute_conditional(step, context)

        agent_prompt = self.create_agent_prompt(step, context)
        response = self.get_completion(agent_prompt, model = self.model, step_counter=step_counter)


        #TODO: add in validation again. Removed to make things faster and help with abation studies
        """
        validation_result = self._validate_response(response, step, step_counter=step_counter)

        if not validation_result['valid']:
            # Try backtracking
            alternative_state = self.execution_state.pop_best_state()
            if alternative_state:
                return self.execute_step(step, alternative_state, step_counter=step_counter)
            else:
                # Need to replan
                return {
                    'success': False,
                    'step_id': step.id,
                    'error': validation_result['reason'],
                    'needs_replan': True
                }
        """

        result = {
            'success': True,
            'step_id': step.id,
            'output': response,
            #'reasoning_trace': validation_result.get('reasoning_trace')
            'reasoning_trace': "N/A"
        }

        # Store successful state
        #self.execution_state.push_state(context | {'current_step_result': result},
        #                             validation_result.get('confidence', 0.5))
        self.execution_state.push_state(context | {'current_step_result': result}, 0.5)

        return result

    def create_agent_prompt(self, step: PlanStep, context: Dict) -> str:
        """Create a prompt incorporating reasoning strategies and chain of thought."""
        #TODO: Optimize this prompting by asking the LLM to reflect on it
        # Build context section showing previous outputs
        context_section = "\n".join([
            f"Step {k.split('_')[1]} Output ({v.get('description', 'N/A')}):"
            f"\n{v.get('output', 'N/A')}\n"
            for k, v in context.items()
            if k.startswith('step_')
        ])

        # Build reasoning strategies section
        #TODO: Fix this reasoning strategies to enable selecting among existing strategies
        #reasoning_section = "\n".join([
        #    self.reasoning_strategies[strategy].prompt_template
        #    for strategy in step.reasoning_strategies
        #])
        reasoning_section = "\n".join([
            strategy for strategy in step.reasoning_strategies ])

        prompt = f"""Task: {step.description}

Available Context:
{context_section}

Input Requirements:
{json.dumps(step.input_spec, indent=2)}

Required Output Format:
{json.dumps(step.output_spec, indent=2)}

Approach this task using these reasoning strategies:
{reasoning_section}

You are a logical reasoning assistant tasked with solving a multi-step logic problem. Please follow these guidelines:

Solve this step by:

1. Break down the approach into smaller, manageable parts. Identify key facts, variables, and constraints.
2. Proceed step-by-step through the problem, explaining your reasoning at each stage. Justify your conclusions with clear logic and evidence from the problem.
3. Consider all possibilities and eliminate any that lead to contradictions. Ensure that each step logically follows from the previous one.
4. Verify your answer
5. Format according to output requirements

Provide your complete thought process and final answer.
"""
        return prompt



    def _execute_loop(self, step: PlanStep, context: Dict) -> Dict:
        """Execute a loop step."""
        results = []
        iterations = 0

        while iterations < step.max_iterations:
            should_continue = self._evaluate_condition(step.condition, context)
            if not should_continue:
                break

            for sub_step in step.sub_steps:
                result = self.execute_step(sub_step, context)
                if not result['success']:
                    return result
                results.append(result)
                context[f'step_{sub_step.id}_output'] = result

            iterations += 1

        return {
            'success': True,
            'step_id': step.id,
            'output': results,
            'iterations': iterations
        }

    def _execute_conditional(self, step: PlanStep, context: Dict) -> Dict:
        """Execute a conditional step."""
        condition_met = self._evaluate_condition(step.condition, context)

        if condition_met and step.sub_steps:
            results = []
            for sub_step in step.sub_steps:
                result = self.execute_step(sub_step, context)
                if not result['success']:
                    return result
                results.append(result)
                context[f'step_{sub_step.id}_output'] = result

            return {
                'success': True,
                'step_id': step.id,
                'output': results,
                'condition_result': True
            }

        return {
            'success': True,
            'step_id': step.id,
            'output': None,
            'condition_result': False
        }

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a condition using the LLM."""
        prompt = f"""Evaluate if this condition is true given the context:

        Condition: {condition}

        Context:
        {json.dumps(context, indent=2)}

        Return ONLY 'True' or 'False'
        """

        response = self.get_completion(prompt, model=self.model)
        return response.strip().lower() == 'true'

    def _validate_response(self, response: str, step: PlanStep, step_counter = None) -> Dict:
        """Validate response using reasoning strategies."""
        validation_prompt = f"""Validate this response:

Response:
{response}

Required Output Format:
{json.dumps(step.output_spec, indent=2)}

Apply these validation strategies:
{json.dumps([s for s in step.reasoning_strategies], indent=2)}

Return JSON with:
{{
    "valid": boolean,
    "reason": string,
    "confidence": number,
    "reasoning_trace": string
}}
"""

        validation_response = self.get_completion(validation_prompt, json = True, model=self.model, step_counter = step_counter)
        return json.loads(validation_response)


    def integrate_outputs(self, execution_results: List[Dict], model = 'gpt-4o-mini') -> str:
        """Integrate all outputs into final answer."""
        prompt = f"""Given these step-by-step results, create a clear final answer:

Results:
{json.dumps(execution_results, indent=2)}

1. Synthesize the results
2. Ensure consistency
3. Provide a clear, direct answer
4. Include key reasoning steps

Return JSON with:
{{
    "key reasoning steps": string,
    "final answer": number
}}
"""

        return self.get_completion(prompt, model= model)
    
class CoTEvaluator:
    def __init__(self, agent):
        """
        Initialize the evaluator with the OpenAI LLM instance.
        """
        self.agent = agent

    def evaluate(self, question, true_response, generated_response):
        """
        Parameters:
        - question (str): The question being answered.
        - true_response (str): The ground-truth, human-verified response.
        - generated_response (str): The machine-generated response.

        Returns:
        - dict: Contains the score and justification.
        """

        prompt = f"""
You are an expert evaluator assessing reasoning quality in math problems. Below is a question, a human-verified solution (True Response), and a machine-generated solution (Generated Response).

**Question**:
{question}

**True Response**:
{true_response}

**Generated Response**:
{generated_response}

**Your Task**:
1. Assign a score from 1 to 10 based on the overall quality of the Generated Response relative to the True Response based on:
   - **Accuracy**: Is the final answer and reasoning correct?
   - **Completeness**: Are all necessary reasoning steps included?
   - **Clarity**: Is the reasoning clear and easy to follow?
   - **Logical Soundness**: Are there any logical errors or unwarranted assumptions?

**Output Format**:
- Accuracy Score: X
- Completeness Score: X
- Clarity Score: X
- Logical Soundness Score: X
"""

        # modify according to agent
        response = self.agent.complete(prompt)
        return response


def parse_and_get_reasoning(input_string):
  try:
    start = input_string.find("```")+4
    end = input_string.find("```", start + 3)

    if start == -1 or end == -1:
        raise ValueError("Triple single quotes not found or improperly formatted.")

    json_content = input_string[start + 3:end]
    data = json.loads(json_content)

    if not isinstance(data, dict):
        raise ValueError("JSON content is not a dictionary.")

    last_value = list(data.values())[-1]

    if not isinstance(last_value, int):
        raise ValueError("The last value is not an integer.")

    return data
  except (json.JSONDecodeError, ValueError) as e:
    return f"Error: {e}"
  

from tqdm import tqdm
cot_evals = []
for i in tqdm(range(len(querys))):
  result = evaluator.evaluate(querys[i], reasoning_patterns[i], parse_and_get_reasoning(reasonings[i]))
  cot_evals.append(result)

print("\nEvaluation Result:")
print(cot_evals)
