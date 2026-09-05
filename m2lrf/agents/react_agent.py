"""
M-2LRF Agent Engine: ReAct Autonomous Agent.
============================================
Implements the Thought -> Action -> Observation -> Final Answer autonomous execution cycle:
1. Formats prompt with tool descriptions and few-shot ReAct examples.
2. Invokes LLM forward generation.
3. Parses action and tool parameters.
4. Executes tool in environment sandbox.
5. Injects observation back into conversational context until final answer.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import re

from m2lrf.agents.tool_calling import ToolCallingEngine


class ReActAgent:
    """
    Autonomous ReAct (Reasoning + Acting) Agent.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        tools: Optional[ToolCallingEngine] = None,
        max_iterations: int = 5,
    ):
        self.llm_fn = llm_fn
        self.tools = tools or ToolCallingEngine()
        self.max_iterations = max_iterations

    def format_system_prompt(self) -> str:
        tools_info = []
        for t in self.tools.tools.values():
            tools_info.append(f"- {t.name}: {t.description} (Args: {list(t.parameters.get('properties', {}).keys())})")
        tools_str = "\n".join(tools_info) if tools_info else "No external tools available."

        return f"""You are an autonomous agent using the ReAct framework.
Available Tools:
{tools_str}

Use the following format:
Thought: think step-by-step about what to do next.
Action: <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>
Observation: result of the action.
... (this Thought/Action/Observation can repeat N times)
Thought: I have enough information to answer the question.
Final Answer: the final response to the user.
"""

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the ReAct loop until 'Final Answer:' is found or max iterations reached.
        """
        trajectory: List[Dict[str, str]] = []
        prompt = f"{self.format_system_prompt()}\n\nUser Question: {user_query}\n"

        for iteration in range(1, self.max_iterations + 1):
            response = self.llm_fn(prompt)
            trajectory.append({"role": "assistant", "content": response})
            prompt += f"{response}\n"

            # Check for Final Answer
            if "Final Answer:" in response:
                final_text = response.split("Final Answer:")[-1].strip()
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "final_answer": final_text,
                    "trajectory": trajectory,
                }

            # Check for Tool Action
            tool_calls = self.tools.parse_tool_calls(response)
            if not tool_calls:
                # No action found, break or return current response
                return {
                    "status": "completed_no_action",
                    "iterations": iteration,
                    "final_answer": response.strip(),
                    "trajectory": trajectory,
                }

            # Execute tool call
            call = tool_calls[0]
            result = self.tools.execute_call(call)
            obs_str = f"Observation: {result.get('output')}\n"
            trajectory.append({"role": "environment", "content": obs_str})
            prompt += obs_str

        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "final_answer": "Execution reached maximum iteration budget.",
            "trajectory": trajectory,
        }
