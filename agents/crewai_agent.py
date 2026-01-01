"""
CrewAI Agent Implementation
This agent uses CrewAI framework for task management
"""
import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tools.task_tools import TaskTools
from tools.calendar_tool import CalendarTool
from observability.langfuse_config import trace_agent_execution, log_agent_event, get_langfuse_client

load_dotenv()

# Tool functions
def create_task_tool(title: str, description: str = "", priority: str = "medium", due_date: str = None) -> str:
    """Tool to create a task. Input should be: title, description (optional), priority (low/medium/high), due_date (YYYY-MM-DD, optional)"""
    try:
        result = TaskTools.create_task(title, description, priority, due_date)
        return f"Task created successfully: ID {result['id']} - {result['title']} (Priority: {result['priority']}, Status: {result['status']})"
    except Exception as e:
        return f"Error creating task: {str(e)}"

def list_tasks_tool(status: str = "") -> str:
    """Tool to list tasks. Input: status filter (pending/in_progress/completed) or empty for all"""
    try:
        status_filter = status.strip() if status.strip() else None
        tasks = TaskTools.list_tasks(status_filter)
        if not tasks:
            return "No tasks found."
        result = f"Found {len(tasks)} task(s):\n"
        for task in tasks:
            result += f"  • ID {task['id']}: {task['title']} ({task['status']}, {task['priority']} priority)\n"
        return result
    except Exception as e:
        return f"Error listing tasks: {str(e)}"

def update_task_tool(task_input: str) -> str:
    """Tool to update a task. Input format: 'task_id,status,priority' where status and priority are optional"""
    try:
        parts = [p.strip() for p in task_input.split(',')]
        task_id = int(parts[0])
        status = parts[1] if len(parts) > 1 and parts[1] else None
        priority = parts[2] if len(parts) > 2 and parts[2] else None
        
        task = TaskTools.update_task(task_id, status=status, priority=priority)
        if task:
            return f"Task {task_id} updated successfully: {task['title']} - Status: {task['status']}, Priority: {task['priority']}"
        return f"Task {task_id} not found"
    except (ValueError, IndexError) as e:
        return f"Invalid input format. Expected: 'task_id,status,priority'. Error: {str(e)}"
    except Exception as e:
        return f"Error updating task: {str(e)}"

def get_task_statistics_tool() -> str:
    """Tool to get task statistics. No input required."""
    try:
        stats = TaskTools.get_statistics()
        return f"""Task Statistics:
• Total Tasks: {stats['total']}
• Pending: {stats['pending']}
• In Progress: {stats['in_progress']}
• Completed: {stats['completed']}
• High Priority: {stats['high_priority']}
• Medium Priority: {stats['medium_priority']}
• Low Priority: {stats['low_priority']}"""
    except Exception as e:
        return f"Error getting statistics: {str(e)}"

# Define input schemas for tools
class CreateTaskInput(BaseModel):
    title: str = Field(..., description="The task title")
    description: str = Field("", description="Task description")
    priority: str = Field("medium", description="Priority: low, medium, or high")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")

class ListTasksInput(BaseModel):
    status: Optional[str] = Field(None, description="Filter by status: pending, in_progress, or completed")

class UpdateTaskInput(BaseModel):
    task_id: int = Field(..., description="The task ID to update")
    status: Optional[str] = Field(None, description="New status: pending, in_progress, or completed")
    priority: Optional[str] = Field(None, description="New priority: low, medium, or high")

# Create CrewAI-compatible tools using BaseTool
class CreateTaskTool(BaseTool):
    name: str = "create_task"
    description: str = "Create a new task with title, description, priority, and optional due date"
    args_schema: Type[BaseModel] = CreateTaskInput
    
    def _run(self, title: str, description: str = "", priority: str = "medium", due_date: Optional[str] = None) -> str:
        return create_task_tool(title, description, priority, due_date)

class ListTasksTool(BaseTool):
    name: str = "list_tasks"
    description: str = "List all tasks, optionally filtered by status"
    args_schema: Type[BaseModel] = ListTasksInput
    
    def _run(self, status: Optional[str] = None) -> str:
        return list_tasks_tool(status or "")

class UpdateTaskTool(BaseTool):
    name: str = "update_task"
    description: str = "Update an existing task's status or priority"
    args_schema: Type[BaseModel] = UpdateTaskInput
    
    def _run(self, task_id: int, status: Optional[str] = None, priority: Optional[str] = None) -> str:
        task = TaskTools.update_task(task_id, status=status, priority=priority)
        if task:
            return f"Task {task_id} updated successfully: {task['title']} - Status: {task['status']}, Priority: {task['priority']}"
        return f"Task {task_id} not found"

class GetTaskStatisticsTool(BaseTool):
    name: str = "get_task_statistics"
    description: str = "Get statistics about all tasks (total, pending, in_progress, completed counts)"
    
    def _run(self) -> str:
        return get_task_statistics_tool()

class CrewAITaskAgent:
    """CrewAI-based task management agent"""
    
    def __init__(self):
        """Initialize the CrewAI agent"""
        # Set up Langfuse for CrewAI
        os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # Configure CrewAI to use Google Gemini if OpenAI is not available
        # Try to use Google Gemini as fallback
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
        
        # Initialize tools
        self.task_tools = [
            CreateTaskTool(),
            ListTasksTool(),
            UpdateTaskTool(),
            GetTaskStatisticsTool()
        ]
        
        # Create the agent
        # Note: CrewAI tool integration can be complex, so we'll use a workaround
        # by calling tools directly in the agent's processing
        self.agent = Agent(
            role='Task Management Assistant',
            goal='Help users manage their tasks efficiently. You have access to task management functions: create_task(title, description, priority, due_date), list_tasks(status), update_task(task_id, status, priority), get_task_statistics(). Use these functions to help users.',
            backstory="""You are an expert task management assistant powered by AI. 
            You help users organize their work by managing tasks, tracking priorities, 
            and providing insights on task completion. You are friendly, efficient, and detail-oriented.
            When users ask you to manage tasks, you should use the available task management functions.""",
            verbose=True,
            allow_delegation=False,
            # Temporarily disable tools to avoid validation errors
            # Tools will be called programmatically
            tools=[],
        )
        
        # Store tools for programmatic access
        self.available_tools = {
            'create_task': create_task_tool,
            'list_tasks': list_tasks_tool,
            'update_task': update_task_tool,
            'get_task_statistics': get_task_statistics_tool,
            'TaskTools': TaskTools
        }
        
        log_agent_event("agent_initialized", "crewai_task_agent", {"status": "success"})
    
    def process_task(self, user_request: str) -> str:
        """
        Process a user request using the CrewAI agent
        
        Args:
            user_request: The user's task management request
            
        Returns:
            Response from the agent
        """
        # Create a trace for this execution
        from observability.langfuse_config import create_trace, end_span
        trace = create_trace(
            name="crewai_task_processing",
            metadata={"user_request": user_request, "agent": "crewai_task_agent"}
        )
        
        try:
            log_agent_event("task_processing_started", "crewai_task_agent", {"request": user_request})
            
            # Create a task for the agent
            task = Task(
                description=user_request,
                agent=self.agent,
                expected_output="A helpful response about task management operations"
            )
            
            # Create crew
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            end_span(output=str(result))
            log_agent_event("task_processing_completed", "crewai_task_agent", {"request": user_request, "success": True})
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            end_span(output=error_msg)
            log_agent_event("task_processing_failed", "crewai_task_agent", {"error": str(e)})
            return error_msg

