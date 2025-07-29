"""
Model Selection Utility
Automatically suggests the best AI model based on task type and content analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from src.config.config import MODEL_OPTIONS

# Task type patterns and their optimal models
TASK_PATTERNS = {
    # Reasoning and complex problem solving
    "reasoning": {
        "patterns": [
            r"solve|calculate|compute|analyze|reason|logic|problem|proof|mathematics?|math|equation",
            r"step.by.step|think|explain why|how does|what is the relationship",
            r"algorithm|optimization|strategy|plan|approach"
        ],
        "models": ["openai/o1-preview", "openai/o1", "openai/o1-mini", "openai/gpt-4o"]
    },
    
    # Code and programming tasks
    "coding": {
        "patterns": [
            r"code|program|script|function|class|debug|refactor|implement",
            r"python|javascript|java|c\+\+|html|css|sql|api|framework",
            r"bug|error|exception|syntax|compile|deploy|test",
            r"```.*```",  # Code blocks
            r"github|repository|pull request|commit"
        ],
        "models": ["openai/gpt-4o", "openai/o1-preview", "openai/gpt-4o-mini"]
    },
    
    # Creative and content generation
    "creative": {
        "patterns": [
            r"write|create|generate|compose|story|poem|article|blog",
            r"creative|imagination|fiction|narrative|character|plot",
            r"lyrics|song|script|dialogue|monologue",
            r"marketing|advertisement|slogan|copy|content"
        ],
        "models": ["openai/gpt-4o", "openai/gpt-4o-mini"]
    },
    
    # Data analysis and research
    "analysis": {
        "patterns": [
            r"analyze|analysis|data|statistics|chart|graph|visualization",
            r"research|study|findings|conclusions|insights|trends",
            r"compare|contrast|evaluate|assess|review|examine",
            r"csv|excel|spreadsheet|dataset|metrics|kpi"
        ],
        "models": ["openai/gpt-4o", "openai/o1-preview", "openai/gpt-4o-mini"]
    },
    
    # Quick questions and general chat
    "general": {
        "patterns": [
            r"^(hi|hello|hey|what|who|when|where|how|why|can you|please|thanks?)",
            r"quick question|simple|brief|short answer|tldr|summary"
        ],
        "models": ["openai/gpt-4o-mini", "openai/gpt-4o"]
    },
    
    # Translation and language tasks
    "language": {
        "patterns": [
            r"translate|translation|language|français|español|deutsch|italiano|中文|日本語|한국어",
            r"grammar|spelling|proofreading|correct|fix|improve writing"
        ],
        "models": ["openai/gpt-4o", "openai/gpt-4o-mini"]
    },
    
    # Image and visual tasks
    "visual": {
        "patterns": [
            r"image|picture|photo|visual|draw|sketch|art|design",
            r"generate image|create image|make picture|visualize"
        ],
        "models": ["openai/gpt-4o", "openai/gpt-4o-mini"]  # For image generation prompts
    }
}

class ModelSelector:
    """Intelligent model selection based on task analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_task_type(self, content: str) -> str:
        """
        Analyze the content to determine the primary task type.
        
        Args:
            content (str): The user's input content
            
        Returns:
            str: The detected task type
        """
        if not content or not isinstance(content, str):
            return "general"
        
        content_lower = content.lower()
        task_scores = {}
        
        # Score each task type based on pattern matches
        for task_type, config in TASK_PATTERNS.items():
            score = 0
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches
            
            # Bonus for longer matches
            if score > 0:
                score += len(content_lower) / 1000  # Small bonus for longer content
            
            task_scores[task_type] = score
        
        # Return the task type with the highest score
        if not task_scores or max(task_scores.values()) == 0:
            return "general"
        
        best_task = max(task_scores, key=task_scores.get)
        self.logger.debug(f"Task analysis: {task_scores}, selected: {best_task}")
        return best_task
    
    def suggest_model(self, content: str, user_preference: Optional[str] = None) -> Tuple[str, str]:
        """
        Suggest the best model for the given content.
        
        Args:
            content (str): The user's input content
            user_preference (Optional[str]): User's preferred model if any
            
        Returns:
            Tuple[str, str]: (suggested_model, reason)
        """
        # If user has a strong preference and it's available, respect it
        if user_preference and user_preference in MODEL_OPTIONS:
            return user_preference, f"Using your preferred model: {user_preference}"
        
        # Analyze the task type
        task_type = self.analyze_task_type(content)
        
        # Get the recommended models for this task type
        if task_type in TASK_PATTERNS:
            recommended_models = TASK_PATTERNS[task_type]["models"]
            
            # Find the first available model from recommendations
            for model in recommended_models:
                if model in MODEL_OPTIONS:
                    reason = f"Optimized for {task_type} tasks"
                    return model, reason
        
        # Fallback to default model
        default_model = "openai/gpt-4o-mini"  # Fast and cost-effective default
        return default_model, "Default model for general tasks"
    
    def get_model_explanation(self, model: str) -> str:
        """
        Get a user-friendly explanation of what the model is best for.
        
        Args:
            model (str): The model name
            
        Returns:
            str: Human-readable explanation
        """
        explanations = {
            "openai/o1-preview": "🧠 Best for complex reasoning, mathematics, and step-by-step problem solving",
            "openai/o1": "🧠 Advanced reasoning model for complex analytical tasks",
            "openai/o1-mini": "⚡ Fast reasoning model for structured problem solving",
            "openai/gpt-4o": "🎯 Balanced model excellent for all tasks including coding, analysis, and creativity",
            "openai/gpt-4o-mini": "⚡ Fast and efficient model for general conversations and quick tasks",
            "openai/gpt-4.1": "💪 Enhanced model with improved capabilities",
            "openai/gpt-4.1-mini": "🚀 Compact version with great performance",
            "openai/gpt-4.1-nano": "⚡ Ultra-fast model for simple tasks",
            "openai/o3-mini": "🔧 Specialized model for focused tasks",
            "openai/o3": "🔬 Advanced model for specialized analysis",
            "openai/o4-mini": "🚀 Next-generation compact model"
        }
        
        return explanations.get(model, f"AI model: {model}")
    
    def suggest_model_with_alternatives(self, content: str, user_preference: Optional[str] = None) -> Dict[str, any]:
        """
        Suggest a model with alternatives and explanations.
        
        Args:
            content (str): The user's input content
            user_preference (Optional[str]): User's preferred model
            
        Returns:
            Dict containing suggestion details
        """
        primary_model, reason = self.suggest_model(content, user_preference)
        task_type = self.analyze_task_type(content)
        
        # Get alternative models for this task
        alternatives = []
        if task_type in TASK_PATTERNS:
            for model in TASK_PATTERNS[task_type]["models"]:
                if model != primary_model and model in MODEL_OPTIONS:
                    alternatives.append({
                        "model": model,
                        "explanation": self.get_model_explanation(model)
                    })
        
        return {
            "suggested_model": primary_model,
            "reason": reason,
            "task_type": task_type,
            "explanation": self.get_model_explanation(primary_model),
            "alternatives": alternatives[:2]  # Limit to 2 alternatives
        }

# Global instance
model_selector = ModelSelector()