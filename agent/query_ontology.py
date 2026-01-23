"""
Query Ontology - Fast keyword and synonym matching for query understanding.

Provides JSON-based ontology for fast query routing before expensive LLM reasoning.
Loads from query_ontology.json file for easy maintenance and updates.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Load ontology from JSON file
ONTOLOGY_FILE = Path(__file__).parent / "query_ontology.json"

def load_ontology() -> Dict[str, Any]:
    """Load query ontology from JSON file."""
    try:
        with open(ONTOLOGY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ontology file: {e}")

# Load ontology at module level
try:
    QUERY_ONTOLOGY = load_ontology()
except Exception as e:
    # Fallback to empty structure if JSON file not found
    import sys
    print(f"[WARNING] Failed to load query ontology: {e}", file=sys.stderr)
    QUERY_ONTOLOGY = {
        "tools": {},
        "query_types": {},
        "cached_queries": {},
        "object_classes": [],
        "semantic_mappings": {}
    }


class QueryOntologyMatcher:
    """
    Fast keyword and synonym-based query matcher.
    
    Uses JSON ontology to quickly route queries before expensive LLM reasoning.
    """
    
    def __init__(self, ontology: Dict[str, Any] = None):
        """
        Initialize ontology matcher.
        
        Args:
            ontology: Query ontology dict (defaults to QUERY_ONTOLOGY)
        """
        self.ontology = ontology or QUERY_ONTOLOGY
        self._build_index()
    
    def _build_index(self):
        """Build fast lookup indexes for keywords and synonyms."""
        # Tool keyword index
        self.tool_keywords = {}
        for tool_name, tool_data in self.ontology["tools"].items():
            keywords = tool_data.get("keywords", [])
            synonyms = tool_data.get("synonyms", {})
            # Flatten synonyms into keyword list
            all_keywords = list(keywords)
            for base_word, syn_list in synonyms.items():
                all_keywords.extend(syn_list)
            self.tool_keywords[tool_name] = [kw.lower() for kw in all_keywords]
        
        # Query type keyword index
        self.query_type_keywords = {}
        for query_type, type_data in self.ontology["query_types"].items():
            keywords = type_data.get("keywords", [])
            self.query_type_keywords[query_type] = [kw.lower() for kw in keywords]
        
        # Phrase index (multi-word patterns for better matching)
        self.phrase_patterns = {}
        cached_queries = self.ontology.get("cached_queries", {})
        for cache_key, cache_data in cached_queries.items():
            prompt = cache_data.get("prompt", "").lower().strip()
            if prompt and len(prompt.split()) > 1:  # Multi-word phrases
                tool = cache_data.get("tool")
                if tool:
                    if tool not in self.phrase_patterns:
                        self.phrase_patterns[tool] = []
                    self.phrase_patterns[tool].append(prompt)
    
    def match_tool(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Match query to a tool using keywords, synonyms, and phrases.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (tool_name, confidence) or None if no match
            Confidence is 0.0-1.0 based on keyword/phrase matches
        """
        query_lower = query.lower()
        
        best_match = None
        best_score = 0.0
        
        # OPTIMIZATION: Check phrases first (higher priority, more accurate)
        for tool_name, phrases in self.phrase_patterns.items():
            for phrase in phrases:
                if phrase in query_lower:
                    # Phrase match = high confidence (0.8-1.0)
                    phrase_score = 0.8 + (0.2 * (len(phrase.split()) / max(len(query_lower.split()), 1)))
                    if phrase_score > best_score:
                        best_score = phrase_score
                        best_match = tool_name
                        break  # Found phrase match, use it
        
        # If no phrase match, check keywords
        if best_score < 0.8:
            for tool_name, keywords in self.tool_keywords.items():
                matches = sum(1 for kw in keywords if kw in query_lower)
                if matches > 0:
                    # Score based on number of keyword matches
                    score = min(matches / max(len(keywords), 1), 1.0)
                    if score > best_score:
                        best_score = score
                        best_match = tool_name
        
        if best_match and best_score > 0.05:  # Lowered threshold from 0.1 to 0.05 (catch more matches)
            return (best_match, best_score)
        return None
    
    def match_query_type(self, query: str) -> Optional[str]:
        """
        Match query to a query type using keywords.
        
        Args:
            query: User query string
            
        Returns:
            Query type string or None
        """
        query_lower = query.lower()
        
        for query_type, keywords in self.query_type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return query_type
        
        return None
    
    def extract_object_class(self, query: str) -> Optional[str]:
        """
        Extract object class from query using ontology.
        
        Args:
            query: User query string
            
        Returns:
            Object class string or None
        """
        query_lower = query.lower()
        
        # Check semantic mappings first (sweater → person)
        for word, mapped_class in self.ontology["semantic_mappings"].items():
            if word.lower() in query_lower:
                return mapped_class
        
        # Check common object classes
        for obj_class in self.ontology["object_classes"]:
            if obj_class.lower() in query_lower:
                return obj_class
        
        return None
    
    def needs_llm_reasoning(self, query: str, matched_tool: Optional[str] = None, confidence: float = 0.0) -> bool:
        """
        Determine if query needs LLM reasoning based on ontology.
        
        Args:
            query: User query string
            matched_tool: Tool matched by keyword matching (if any)
            confidence: Confidence score from tool matching (0.0-1.0)
            
        Returns:
            True if LLM reasoning is required, False if keyword matching is sufficient
        """
        # OPTIMIZATION: High confidence matches (>0.3) always skip LLM
        # This catches most common queries without LLM delay
        if confidence > 0.3:
            return False
        
        query_type = self.match_query_type(query)
        
        if query_type:
            type_data = self.ontology["query_types"].get(query_type, {})
            if type_data.get("requires_llm", False):
                return True
        
        # OPTIMIZATION: Color queries only need LLM if semantic mapping needed
        # Check if query contains semantic mapping keywords (e.g., "sweater", "shirt")
        semantic_keywords = ["sweater", "shirt", "jacket", "pants", "dress", "face", "hand", "foot", "head"]
        has_semantic_keyword = any(kw in query.lower() for kw in semantic_keywords)
        
        if matched_tool == "color_detection":
            # Only need LLM if semantic mapping required
            return has_semantic_keyword
        
        # Spatial queries need LLM for relationship understanding
        if query_type == "spatial":
            return True
        
        # Environment queries need LLM for object combination analysis
        if query_type == "environment":
            return True
        
        # OPTIMIZATION: If we have tool OR query_type, skip LLM (unless explicitly required)
        # This is more aggressive - trust ontology matches
        if matched_tool or query_type:
            return False  # Trust ontology match, skip LLM
        
        # If we couldn't match anything, might need LLM
        return True
    
    def match_cached_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Match query to cached query definitions (for quick prompt buttons).
        
        Args:
            query: User query string
            
        Returns:
            Dict with cached query info or None if no match
        """
        query_lower = query.lower().strip()
        cached_queries = self.ontology.get("cached_queries", {})
        
        # Check exact prompt matches first
        for cache_key, cache_data in cached_queries.items():
            cached_prompt = cache_data.get("prompt", "").lower().strip()
            if cached_prompt == query_lower:
                return {
                    "tool": cache_data.get("tool"),
                    "query_type": cache_data.get("query_type"),
                    "object_class": None,
                    "confidence": 1.0,
                    "needs_llm": cache_data.get("needs_llm", False),
                    "direct_call": cache_data.get("direct_call", False),
                    "cached_query": cache_key,
                    "reasoning": f"cached_query:{cache_key}"
                }
        
        # Check keyword matches in cached query prompts
        for cache_key, cache_data in cached_queries.items():
            cached_prompt = cache_data.get("prompt", "").lower()
            # Check if query contains key phrases from cached prompt
            prompt_words = set(cached_prompt.split())
            query_words = set(query_lower.split())
            # If 50%+ of prompt words match, consider it a match
            if len(prompt_words) > 0:
                match_ratio = len(prompt_words & query_words) / len(prompt_words)
                if match_ratio >= 0.5:
                    return {
                        "tool": cache_data.get("tool"),
                        "query_type": cache_data.get("query_type"),
                        "object_class": None,
                        "confidence": match_ratio,
                        "needs_llm": cache_data.get("needs_llm", False),
                        "direct_call": cache_data.get("direct_call", False),
                        "cached_query": cache_key,
                        "reasoning": f"cached_query:{cache_key}"
                    }
        
        return None
    
    def quick_route(self, query: str) -> Dict[str, Any]:
        """
        Fast keyword-based routing without LLM.
        
        Args:
            query: User query string
            
        Returns:
            Dict with tool, query_type, object_class, and needs_llm flag
        """
        # Check cached queries first (for quick prompt buttons)
        cached_query = self.match_cached_query(query)
        if cached_query:
            return cached_query
        
        tool_match = self.match_tool(query)
        tool_name = tool_match[0] if tool_match else None
        confidence = tool_match[1] if tool_match else 0.0
        
        query_type = self.match_query_type(query)
        object_class = self.extract_object_class(query)
        
        # Pass confidence to needs_llm_reasoning for high-confidence bypass
        needs_llm = self.needs_llm_reasoning(query, tool_name, confidence)
        
        return {
            "tool": tool_name,
            "query_type": query_type or "unknown",
            "object_class": object_class,
            "confidence": confidence,
            "needs_llm": needs_llm,
            "reasoning": "keyword_match" if tool_match else "no_match"
        }

