"""
Google ADK (Agent Development Kit) Agent Implementation
This agent uses Google's Gemini API for task management
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
from tools.task_tools import TaskTools
from tools.calendar_tool import CalendarTool
from observability.langfuse_config import trace_agent_execution, log_agent_event, get_langfuse_client
from typing import Dict, Any
import json

load_dotenv()

class GoogleADKAgent:
    """Google ADK-based task management agent using Gemini"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Google ADK agent
        
        Args:
            model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        """
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env file")
        
        genai.configure(api_key=api_key)
        
        # Try to use the model name, or fallback to available models
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception:
            # If model name fails, try with models/ prefix or use default
            try:
                if not model_name.startswith("models/"):
                    self.model = genai.GenerativeModel(f"models/{model_name}")
                else:
                    # Try gemini-2.5-flash as fallback
                    self.model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                # Last resort: use gemini-2.5-flash
                self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.task_tools = TaskTools()
        self.calendar_tool = CalendarTool() if os.path.exists("credentials.json") else None
        
        log_agent_event("agent_initialized", "google_adk_agent", {"model": model_name, "status": "success"})
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent"""
        return """You are a helpful task management assistant powered by Google's Gemini AI.
You can help users with the following operations:
1. Create tasks with title, description, priority (low/medium/high), and optional due date
2. List tasks, optionally filtered by status (pending/in_progress/completed)
3. Update task status or priority
4. Get task statistics
5. Create calendar events from tasks (if calendar integration is available)

When users make requests, analyze their intent and use the appropriate functions.
Always provide clear, helpful responses."""
    
    def _parse_user_request(self, user_request: str) -> Dict[str, Any]:
        """
        Parse user request and determine the action
        
        Args:
            user_request: User's request text
            
        Returns:
            Dictionary with action type and parameters
        """
        request_lower = user_request.lower()
        
        # Pattern matching for different actions
        if any(word in request_lower for word in ['create', 'add', 'new task']):
            return {"action": "create", "request": user_request}
        elif any(word in request_lower for word in ['list', 'show', 'get tasks', 'tasks']):
            return {"action": "list", "request": user_request}
        elif any(word in request_lower for word in ['update', 'change', 'modify', 'mark']):
            return {"action": "update", "request": user_request}
        elif any(word in request_lower for word in ['statistics', 'stats', 'summary', 'overview']):
            return {"action": "statistics", "request": user_request}
        elif any(word in request_lower for word in ['delete', 'remove']):
            return {"action": "delete", "request": user_request}
        else:
            return {"action": "general", "request": user_request}
    
    def _extract_task_info(self, user_request: str) -> Dict[str, Any]:
        """
        Extract task information from user request using Gemini
        
        Args:
            user_request: User's request text
            
        Returns:
            Dictionary with extracted task information
        """
        extraction_prompt = f"""Extract task information from the following user request: "{user_request}"

Return a JSON object with the following fields if available:
- title: The task title
- description: The task description
- priority: low, medium, or high (default: medium)
- due_date: Due date in YYYY-MM-DD format if mentioned
- task_id: Task ID if mentioned (for updates/deletes)
- status: pending, in_progress, or completed (for updates)

Only include fields that are mentioned in the request. Return only valid JSON."""

        try:
            response = self.model.generate_content(extraction_prompt)
            # Try to extract JSON from response
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(response_text)
        except Exception as e:
            log_agent_event("task_info_extraction_failed", "google_adk_agent", {"error": str(e)})
            return {}
    
    def _execute_action(self, action: str, params: Dict[str, Any]) -> str:
        """
        Execute the determined action
        
        Args:
            action: Action type (create, list, update, delete, statistics)
            params: Parameters for the action
            
        Returns:
            Result of the action
        """
        try:
            if action == "create":
                title = params.get("title", "Untitled Task")
                description = params.get("description", "")
                priority = params.get("priority", "medium")
                due_date = params.get("due_date")
                
                task = self.task_tools.create_task(title, description, priority, due_date)
                return f"✅ Task created successfully!\nID: {task['id']}\nTitle: {task['title']}\nPriority: {task['priority']}\nStatus: {task['status']}"
            
            elif action == "list":
                status = params.get("status")
                tasks = self.task_tools.list_tasks(status)
                
                if not tasks:
                    return "No tasks found."
                
                result = f"📋 Found {len(tasks)} task(s):\n\n"
                for task in tasks:
                    result += f"• ID {task['id']}: {task['title']} ({task['status']}, {task['priority']} priority)\n"
                    if task.get('due_date'):
                        result += f"  Due: {task['due_date']}\n"
                    if task.get('description'):
                        result += f"  Description: {task['description']}\n"
                    result += "\n"
                
                return result
            
            elif action == "update":
                task_id = params.get("task_id")
                if not task_id:
                    return "Error: Task ID is required for updates."
                
                task = self.task_tools.update_task(
                    task_id,
                    status=params.get("status"),
                    priority=params.get("priority"),
                    title=params.get("title"),
                    description=params.get("description")
                )
                
                if task:
                    return f"✅ Task {task_id} updated successfully!\nTitle: {task['title']}\nStatus: {task['status']}\nPriority: {task['priority']}"
                else:
                    return f"❌ Task {task_id} not found."
            
            elif action == "delete":
                task_id = params.get("task_id")
                if not task_id:
                    return "Error: Task ID is required for deletion."
                
                if self.task_tools.delete_task(task_id):
                    return f"✅ Task {task_id} deleted successfully!"
                else:
                    return f"❌ Task {task_id} not found."
            
            elif action == "statistics":
                stats = self.task_tools.get_statistics()
                return f"""📊 Task Statistics:
• Total Tasks: {stats['total']}
• Pending: {stats['pending']}
• In Progress: {stats['in_progress']}
• Completed: {stats['completed']}
• High Priority: {stats['high_priority']}
• Medium Priority: {stats['medium_priority']}
• Low Priority: {stats['low_priority']}"""
            
            else:
                return "I understand your request, but I'm not sure how to handle it. Please try:\n- Creating a task\n- Listing tasks\n- Updating a task\n- Getting statistics"
        
        except Exception as e:
            log_agent_event("action_execution_failed", "google_adk_agent", {"action": action, "error": str(e)})
            return f"Error executing action: {str(e)}"
    
    def process_request(self, user_request: str) -> str:
        """
        Process a user request using Google ADK
        
        Args:
            user_request: The user's task management request
            
        Returns:
            Response from the agent
        """
        # Create a trace for this execution
        from observability.langfuse_config import create_trace, end_span
        trace = create_trace(
            name="google_adk_task_processing",
            metadata={"user_request": user_request, "agent": "google_adk_agent"}
        )
        
        try:
            log_agent_event("task_processing_started", "google_adk_agent", {"request": user_request})
            
            # Parse the request
            parsed = self._parse_user_request(user_request)
            action = parsed["action"]
            
            # Extract parameters if needed
            params = {}
            if action in ["create", "update", "delete"]:
                params = self._extract_task_info(user_request)
            
            # Execute the action
            result = self._execute_action(action, params)
            
            # Generate a natural language response using Gemini
            response_prompt = f"""Based on the task management operation result below, provide a friendly, natural response to the user.

Operation Result:
{result}

User's Original Request:
{user_request}

Provide a concise, helpful response that acknowledges what was done."""

            response = self.model.generate_content(response_prompt)
            final_result = response.text
            
            end_span(output=final_result)
            log_agent_event("task_processing_completed", "google_adk_agent", {"request": user_request, "success": True})
            
            return final_result
        
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            end_span(output=error_msg)
            log_agent_event("task_processing_failed", "google_adk_agent", {"error": str(e)})
            return error_msg

