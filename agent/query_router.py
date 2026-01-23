"""
Query Understanding Router - Routes queries to appropriate tools using LLM reasoning.

Uses prompt templates to understand semantic relationships and select the right tool.
"""
import asyncio
from typing import Dict, Any, Optional
from agent.prompt_templates import (
    QUERY_ROUTER_TEMPLATE,
    OBJECT_DETECTION_TEMPLATE,
    COLOR_DETECTION_TEMPLATE,
    DISTANCE_SPATIAL_TEMPLATE,
    TRACKING_TEMPLATE,
    RESPONSE_FORMATTING_TEMPLATE,
    SOF_OPERATIONAL_TEMPLATE,
    VLM_RESPONSE_TEMPLATE,
    parse_router_response,
    parse_detection_response,
    parse_color_response,
    parse_distance_response,
    parse_tracking_response
)


class QueryUnderstandingRouter:
    """
    Routes user queries to appropriate tools using LLM-based semantic understanding.
    
    Handles:
    - Object part mapping (sweater → person)
    - Spatial relationships (wrt, relative to)
    - Distance comparisons
    - Environment inference
    """
    
    def __init__(self, llm, verbose: bool = False):
        """
        Initialize query router.
        
        Args:
            llm: LLM instance (DockerLLMAdapter or LlamaCpp)
            verbose: Print debug messages
        """
        self.llm = llm
        self.verbose = verbose
    
    async def validate_tool_choice(self, query: str, suggested_tool: str, suggested_object_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Lightweight LLM validation of ontology-suggested tool choice.
        
        This maintains agentic nature by having LLM confirm/override tool selection,
        but uses a shorter, faster prompt than full reasoning.
        
        Args:
            query: User query string
            suggested_tool: Tool suggested by ontology matching
            suggested_object_class: Object class suggested by ontology (if any)
            
        Returns:
            Dict with validated tool, object_class, and confidence
        """
        validation_prompt = f"""Validate this tool choice for the user query.

User query: "{query}"
Ontology suggested tool: {suggested_tool}
Suggested object class: {suggested_object_class or "none"}

Respond with ONLY valid JSON:
{{
  "tool": "{suggested_tool}" or different tool name,
  "object_class": "{suggested_object_class}" or different class or null,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation (1 sentence max)"
}}

If the suggested tool is correct, return it. If not, suggest the correct tool."""
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(validation_prompt)
            else:
                response = await asyncio.to_thread(self.llm.invoke, validation_prompt)
                if isinstance(response, str):
                    pass
                else:
                    response = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            try:
                parsed = json.loads(response.strip())
                if self.verbose:
                    print(f"[QueryRouter] Validation result: {parsed}")
                return parsed
            except json.JSONDecodeError:
                # Fallback: trust ontology suggestion
                if self.verbose:
                    print(f"[QueryRouter] Validation response not JSON, trusting ontology suggestion")
                return {
                    "tool": suggested_tool,
                    "object_class": suggested_object_class,
                    "confidence": 0.7,
                    "reasoning": "LLM validation failed, using ontology suggestion"
                }
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Validation error: {e}, trusting ontology suggestion")
            return {
                "tool": suggested_tool,
                "object_class": suggested_object_class,
                "confidence": 0.7,
                "reasoning": f"Validation error: {str(e)}, using ontology suggestion"
            }
    
    async def understand_query(self, query: str, detection_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Understand user query and determine tool/parameters.
        
        Args:
            query: User query string
            detection_context: Optional dict with current frame detections to inform routing
            
        Returns:
            Dict with tool, object_class, query_type, parameters, etc.
        """
        if self.verbose:
            print(f"[QueryRouter] Understanding query: {query}")
            if detection_context:
                detections = detection_context.get("detections", [])
                print(f"[QueryRouter] Detection context: {len(detections)} objects detected")
        
        # Format router prompt with detection context if available
        if detection_context:
            detections = detection_context.get("detections", [])
            if detections:
                # Build context string
                class_counts = {}
                for det in detections:
                    cls = det.get("class", "unknown")
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                context_str = f"\n\nCURRENT VIDEO FRAME CONTEXT:\n"
                context_str += f"The following objects are currently detected in the video feed:\n"
                for cls, count in sorted(class_counts.items()):
                    context_str += f"  - {cls}: {count}\n"
                context_str += f"\nUse this context to better understand the user's query and route to the appropriate tool."
                
                # Append context to query
                query_with_context = query + context_str
            else:
                query_with_context = query + "\n\nCURRENT VIDEO FRAME CONTEXT: No objects currently detected in the video feed."
        else:
            query_with_context = query
        
        # Format router prompt
        router_prompt = QUERY_ROUTER_TEMPLATE.format_messages(query=query_with_context)
        
        # Extract system and user messages
        system_msg = None
        user_msg = None
        for msg in router_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        # Call LLM with system prompt
        try:
            if hasattr(self.llm, '_acall'):
                # DockerLLMAdapter or async LLM
                response = await self.llm._acall(
                    user_msg,
                    system_prompt=system_msg
                )
            else:
                # Sync LLM (LlamaCpp)
                response = await asyncio.to_thread(
                    self.llm.invoke,
                    router_prompt
                )
                if isinstance(response, str):
                    pass  # Already a string
                else:
                    response = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            parsed = parse_router_response(response)
            
            if self.verbose:
                print(f"[QueryRouter] Parsed: {parsed}")
            
            return parsed
            
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Error: {e}")
            # Fallback to default
            return {
                "tool": None,
                "object_class": None,
                "secondary_object_class": None,
                "query_type": "unknown",
                "reasoning": f"Error: {str(e)}",
                "parameters": {}
            }
    
    async def understand_detection_query(self, query: str) -> Dict[str, Any]:
        """Understand object detection query with semantic mapping."""
        detection_prompt = OBJECT_DETECTION_TEMPLATE.format_messages(query=query)
        
        system_msg = None
        user_msg = None
        for msg in detection_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(user_msg, system_prompt=system_msg)
            else:
                response = await asyncio.to_thread(self.llm.invoke, detection_prompt)
                if not isinstance(response, str):
                    response = response.content if hasattr(response, 'content') else str(response)
            
            return parse_detection_response(response)
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Detection query error: {e}")
            return {"object_class": None, "is_count_query": False, "is_environment_query": False}
    
    async def understand_color_query(self, query: str) -> Dict[str, Any]:
        """Understand color detection query with object part mapping."""
        color_prompt = COLOR_DETECTION_TEMPLATE.format_messages(query=query)
        
        system_msg = None
        user_msg = None
        for msg in color_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(user_msg, system_prompt=system_msg)
            else:
                response = await asyncio.to_thread(self.llm.invoke, color_prompt)
                if not isinstance(response, str):
                    response = response.content if hasattr(response, 'content') else str(response)
            
            return parse_color_response(response)
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Color query error: {e}")
            return {"object_class": None, "region_type": "whole", "specific_part": None}
    
    async def understand_distance_query(self, query: str) -> Dict[str, Any]:
        """Understand distance/spatial query."""
        distance_prompt = DISTANCE_SPATIAL_TEMPLATE.format_messages(query=query)
        
        system_msg = None
        user_msg = None
        for msg in distance_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(user_msg, system_prompt=system_msg)
            else:
                response = await asyncio.to_thread(self.llm.invoke, distance_prompt)
                if not isinstance(response, str):
                    response = response.content if hasattr(response, 'content') else str(response)
            
            return parse_distance_response(response)
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Distance query error: {e}")
            return {"primary_object": None, "reference_object": None, "query_type": "distance"}
    
    async def understand_tracking_query(self, query: str) -> Dict[str, Any]:
        """Understand tracking query."""
        tracking_prompt = TRACKING_TEMPLATE.format_messages(query=query)
        
        system_msg = None
        user_msg = None
        for msg in tracking_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(user_msg, system_prompt=system_msg)
            else:
                response = await asyncio.to_thread(self.llm.invoke, tracking_prompt)
                if not isinstance(response, str):
                    response = response.content if hasattr(response, 'content') else str(response)
            
            return parse_tracking_response(response)
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Tracking query error: {e}")
            return {"object_class": None, "track_id": None, "query_type": "tracking"}
    
    async def format_response(self, user_query: str, tool_results: str) -> str:
        """Format tool results into natural language response."""
        format_prompt = RESPONSE_FORMATTING_TEMPLATE.format_messages(
            user_query=user_query,
            tool_results=tool_results
        )
        
        system_msg = None
        user_msg = None
        for msg in format_prompt:
            if msg.type == "system":
                system_msg = msg.content
            elif msg.type == "human":
                user_msg = msg.content
        
        try:
            if hasattr(self.llm, '_acall'):
                response = await self.llm._acall(user_msg, system_prompt=system_msg)
            else:
                response = await asyncio.to_thread(self.llm.invoke, format_prompt)
                if not isinstance(response, str):
                    response = response.content if hasattr(response, 'content') else str(response)
            
            # Enforce 20-word limit
            response = response.strip()
            words = response.split()
            if len(words) > 20:
                response = ' '.join(words[:20]) + "..."
            
            return response
        except Exception as e:
            if self.verbose:
                print(f"[QueryRouter] Format response error: {e}")
            # Fallback to raw tool results (also truncate)
            words = tool_results.split()
            if len(words) > 20:
                return ' '.join(words[:20]) + "..."
            return tool_results

