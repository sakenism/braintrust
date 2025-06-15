"""
Braintrust Platform SDK Class
A comprehensive Python class for interacting with Braintrust's AI development platform.

Features covered:
- Experiments and evaluations
- Datasets management  
- Prompts and versioning
- AI Proxy for multiple providers
- Logging and monitoring
- Functions and custom scorers
- Playgrounds and testing
- Organizations and access control
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class CacheMode(Enum):
    AUTO = "auto"
    ALWAYS = "always" 
    NEVER = "never"


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AWS = "aws"
    MISTRAL = "mistral"
    TOGETHER = "together"


@dataclass
class BraintrustConfig:
    """Configuration for Braintrust client"""
    api_key: str
    base_url: str = "https://api.braintrust.dev/v1"
    proxy_url: str = "https://api.braintrust.dev/v1/proxy"
    org_name: Optional[str] = None
    timeout: int = 30


@dataclass 
class ExperimentResult:
    """Results from running an experiment"""
    id: str
    name: str
    project_id: str
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    permalink: str


@dataclass
class DatasetRecord:
    """A single record in a dataset"""
    id: str
    input: Any
    expected: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class BraintrustSDK:
    """
    Comprehensive SDK for Braintrust AI development platform
    
    Main capabilities:
    1. Experiments and Evaluations
    2. Dataset Management
    3. Prompt Management
    4. AI Proxy for Multiple Providers
    5. Logging and Monitoring
    6. Custom Functions and Scorers
    7. Project and Organization Management
    """
    
    def __init__(self, config: BraintrustConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })
        
        if config.org_name:
            self.session.headers.update({"x-bt-org-name": config.org_name})
    
    # ============ PROJECT MANAGEMENT ============
    
    def create_project(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new project"""
        payload = {"name": name}
        if description:
            payload["description"] = description
            
        response = self.session.post(f"{self.config.base_url}/project", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_project(self, project_name: str) -> Dict[str, Any]:
        """Get project by name"""
        response = self.session.get(f"{self.config.base_url}/project", 
                                   params={"project_name": project_name})
        response.raise_for_status()
        return response.json()
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        response = self.session.get(f"{self.config.base_url}/project")
        response.raise_for_status()
        return response.json().get("objects", [])
    
    # ============ EXPERIMENT MANAGEMENT ============
    
    def create_experiment(self, project_name: str, experiment_name: str, 
                         description: Optional[str] = None,
                         base_exp_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new experiment"""
        # Get or create project
        try:
            project = self.get_project(project_name)
        except:
            project = self.create_project(project_name)
        
        payload = {
            "name": experiment_name,
            "project_id": project["id"]
        }
        
        if description:
            payload["description"] = description
        if base_exp_id:
            payload["base_exp_id"] = base_exp_id
            
        response = self.session.post(f"{self.config.base_url}/experiment", json=payload)
        response.raise_for_status()
        return response.json()
    
    def log_experiment_event(self, experiment_id: str, 
                           input_data: Any,
                           output_data: Any,
                           expected: Optional[Any] = None,
                           scores: Optional[Dict[str, float]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None) -> str:
        """Log a single event to an experiment"""
        event = {
            "id": uuid.uuid4().hex,
            "input": input_data,
            "output": output_data
        }
        
        if expected is not None:
            event["expected"] = expected
        if scores:
            event["scores"] = scores
        if metadata:
            event["metadata"] = metadata
        if tags:
            event["tags"] = tags
            
        response = self.session.post(
            f"{self.config.base_url}/experiment/{experiment_id}/insert",
            json={"events": [event]}
        )
        response.raise_for_status()
        return event["id"]
    
    def run_evaluation(self, project_name: str, eval_name: str,
                      data: List[Dict[str, Any]],
                      task_function: Callable,
                      scorers: List[Union[str, Callable]],
                      metadata: Optional[Dict[str, Any]] = None) -> ExperimentResult:
        """Run a complete evaluation"""
        experiment = self.create_experiment(project_name, eval_name)
        experiment_id = experiment["id"]
        
        results = []
        for item in data:
            # Run task function
            try:
                output = task_function(item.get("input"))
                
                # Calculate scores using provided scorers
                scores = {}
                for scorer in scorers:
                    if callable(scorer):
                        score_result = scorer(output=output, expected=item.get("expected"), 
                                           input=item.get("input"))
                        if isinstance(score_result, dict):
                            scores.update(score_result)
                        else:
                            scores[getattr(scorer, '__name__', 'custom_score')] = score_result
                    else:
                        # Handle string scorer names
                        scores[scorer] = self._calculate_builtin_score(scorer, output, item.get("expected"))
                
                # Log to experiment
                event_id = self.log_experiment_event(
                    experiment_id,
                    item.get("input"),
                    output,
                    item.get("expected"),
                    scores,
                    item.get("metadata")
                )
                
                results.append({
                    "event_id": event_id,
                    "input": item.get("input"),
                    "output": output,
                    "scores": scores
                })
                
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        # Generate summary scores
        summary_scores = self._calculate_summary_scores(results)
        
        return ExperimentResult(
            id=experiment_id,
            name=eval_name,
            project_id=experiment["project_id"],
            scores=summary_scores,
            metadata=metadata or {},
            permalink=f"https://www.braintrust.dev/app/experiments/{experiment_id}"
        )
    
    # ============ DATASET MANAGEMENT ============
    
    def create_dataset(self, project_name: str, dataset_name: str,
                      description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new dataset"""
        # Get or create project
        try:
            project = self.get_project(project_name)
        except:
            project = self.create_project(project_name)
        
        payload = {
            "name": dataset_name,
            "project_id": project["id"]
        }
        
        if description:
            payload["description"] = description
            
        response = self.session.post(f"{self.config.base_url}/dataset", json=payload)
        response.raise_for_status()
        return response.json()
    
    def insert_dataset_records(self, dataset_id: str, 
                             records: List[DatasetRecord]) -> List[str]:
        """Insert multiple records into a dataset"""
        events = []
        for record in records:
            event = {
                "id": record.id or uuid.uuid4().hex,
                "input": record.input
            }
            
            if record.expected is not None:
                event["expected"] = record.expected
            if record.metadata:
                event["metadata"] = record.metadata
            if record.tags:
                event["tags"] = record.tags
                
            events.append(event)
        
        response = self.session.post(
            f"{self.config.base_url}/dataset/{dataset_id}/insert",
            json={"events": events}
        )
        response.raise_for_status()
        return [event["id"] for event in events]
    
    def get_dataset(self, project_name: str, dataset_name: str) -> Dict[str, Any]:
        """Get dataset by name"""
        response = self.session.get(
            f"{self.config.base_url}/dataset",
            params={"project_name": project_name, "dataset_name": dataset_name}
        )
        response.raise_for_status()
        return response.json()
    
    # ============ PROMPT MANAGEMENT ============
    
    def create_prompt(self, name: str, slug: str, prompt_data: Dict[str, Any],
                     description: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a versioned prompt"""
        payload = {
            "name": name,
            "slug": slug,
            "prompt_data": prompt_data
        }
        
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
            
        response = self.session.post(f"{self.config.base_url}/prompt", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_prompt(self, prompt_slug: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get prompt by slug and optional version"""
        params = {"slug": prompt_slug}
        if version:
            params["version"] = version
            
        response = self.session.get(f"{self.config.base_url}/prompt", params=params)
        response.raise_for_status()
        return response.json()
    
    def update_prompt(self, prompt_id: str, prompt_data: Dict[str, Any],
                     description: Optional[str] = None) -> Dict[str, Any]:
        """Update an existing prompt"""
        payload = {"prompt_data": prompt_data}
        if description:
            payload["description"] = description
            
        response = self.session.patch(f"{self.config.base_url}/prompt/{prompt_id}", 
                                    json=payload)
        response.raise_for_status()
        return response.json()
    
    # ============ AI PROXY FUNCTIONALITY ============
    
    def create_proxy_client(self, provider: ModelProvider = ModelProvider.OPENAI,
                          api_key: Optional[str] = None) -> Dict[str, Any]:
        """Create configuration for AI proxy client"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Use Braintrust API key for unified access or provider-specific key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            
        if self.config.org_name:
            headers["x-bt-org-name"] = self.config.org_name
            
        return {
            "base_url": self.config.proxy_url,
            "headers": headers,
            "provider": provider.value
        }
    
    def proxy_chat_completion(self, model: str, messages: List[Dict[str, str]],
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            cache_mode: CacheMode = CacheMode.AUTO,
                            cache_ttl: Optional[int] = None,
                            seed: Optional[int] = None,
                            provider: ModelProvider = ModelProvider.OPENAI) -> Dict[str, Any]:
        """Make a chat completion request through the AI proxy"""
        headers = self.session.headers.copy()
        headers[f"x-bt-use-cache"] = cache_mode.value
        
        if cache_ttl:
            headers["x-bt-cache-ttl"] = str(cache_ttl)
            
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if seed:
            payload["seed"] = seed
            
        response = self.session.post(
            f"{self.config.proxy_url}/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    # ============ LOGGING AND MONITORING ============
    
    def create_logger(self, project_name: str) -> Dict[str, Any]:
        """Create a logger for production monitoring"""
        try:
            project = self.get_project(project_name)
        except:
            project = self.create_project(project_name)
            
        return {
            "project_id": project["id"],
            "project_name": project_name,
            "logger_id": uuid.uuid4().hex
        }
    
    def log_production_event(self, project_name: str,
                           input_data: Any,
                           output_data: Any,
                           model: Optional[str] = None,
                           latency: Optional[float] = None,
                           tokens_used: Optional[int] = None,
                           cost: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a production event for monitoring"""
        try:
            project = self.get_project(project_name)
        except:
            project = self.create_project(project_name)
        
        event = {
            "id": uuid.uuid4().hex,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if model:
            event["model"] = model
        if latency:
            event["latency"] = latency
        if tokens_used:
            event["tokens_used"] = tokens_used
        if cost:
            event["cost"] = cost
        if metadata:
            event["metadata"] = metadata
            
        response = self.session.post(
            f"{self.config.base_url}/logs/{project['id']}/insert",
            json={"events": [event]}
        )
        response.raise_for_status()
        return event["id"]
    
    # ============ FUNCTIONS AND SCORERS ============
    
    def create_function(self, name: str, slug: str, 
                       function_data: Dict[str, Any],
                       function_type: str = "scorer",
                       description: Optional[str] = None) -> Dict[str, Any]:
        """Create a custom function (scorer or tool)"""
        payload = {
            "name": name,
            "slug": slug,
            "function_data": function_data,
            "function_type": function_type
        }
        
        if description:
            payload["description"] = description
            
        response = self.session.post(f"{self.config.base_url}/function", json=payload)
        response.raise_for_status()
        return response.json()
    
    def invoke_function(self, function_slug: str, input_data: Any,
                       version: Optional[str] = None) -> Any:
        """Invoke a custom function"""
        payload = {
            "input": input_data
        }
        
        params = {"slug": function_slug}
        if version:
            params["version"] = version
            
        response = self.session.post(
            f"{self.config.base_url}/function/invoke",
            json=payload,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # ============ ORGANIZATION AND ACCESS CONTROL ============
    
    def get_organization(self) -> Dict[str, Any]:
        """Get current organization details"""
        response = self.session.get(f"{self.config.base_url}/organization")
        response.raise_for_status()
        return response.json()
    
    def create_api_key(self, name: str, 
                      permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new API key"""
        payload = {"name": name}
        if permissions:
            payload["permissions"] = permissions
            
        response = self.session.post(f"{self.config.base_url}/api_key", json=payload)
        response.raise_for_status()
        return response.json()
    
    def manage_user_access(self, user_email: str, role: str,
                          project_id: Optional[str] = None) -> Dict[str, Any]:
        """Add user to organization or project with specific role"""
        payload = {
            "user_email": user_email,
            "role": role
        }
        
        if project_id:
            payload["project_id"] = project_id
            
        response = self.session.post(f"{self.config.base_url}/acl", json=payload)
        response.raise_for_status()
        return response.json()
    
    # ============ UTILITY METHODS ============
    
    def _calculate_builtin_score(self, scorer_name: str, output: Any, expected: Any) -> float:
        """Calculate score using built-in scorers"""
        if scorer_name.lower() == "exact_match":
            return 1.0 if output == expected else 0.0
        elif scorer_name.lower() == "contains":
            return 1.0 if str(expected).lower() in str(output).lower() else 0.0
        else:
            # Default to similarity-based scoring
            return 0.5  # Placeholder
    
    def _calculate_summary_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary statistics from evaluation results"""
        if not results:
            return {}
            
        summary = {}
        all_score_names = set()
        
        for result in results:
            all_score_names.update(result.get("scores", {}).keys())
        
        for score_name in all_score_names:
            scores = [r.get("scores", {}).get(score_name, 0) for r in results 
                     if score_name in r.get("scores", {})]
            if scores:
                summary[score_name] = sum(scores) / len(scores)
                summary[f"{score_name}_min"] = min(scores)
                summary[f"{score_name}_max"] = max(scores)
        
        return summary
    
    def get_experiment_permalink(self, experiment_id: str) -> str:
        """Generate permalink to experiment in Braintrust UI"""
        return f"https://www.braintrust.dev/app/experiments/{experiment_id}"
    
    def get_project_permalink(self, project_id: str) -> str:
        """Generate permalink to project in Braintrust UI"""
        return f"https://www.braintrust.dev/app/projects/{project_id}"


# ============ USAGE EXAMPLE ============

def example_usage():
    """Example of how to use the BraintrustSDK"""
    
    # Initialize SDK
    config = BraintrustConfig(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        org_name="My Organization"
    )
    bt = BraintrustSDK(config)
    
    # Create a project
    project = bt.create_project("My AI App", "Testing my AI application")
    
    # Create a dataset
    dataset = bt.create_dataset("My AI App", "test_dataset")
    dataset_records = [
        DatasetRecord(id="1", input="Hello", expected="Hi there!"),
        DatasetRecord(id="2", input="Goodbye", expected="See you later!")
    ]
    bt.insert_dataset_records(dataset["id"], dataset_records)
    
    # Define a simple task function
    def my_task(input_text):
        return f"Response to: {input_text}"
    
    # Define a custom scorer
    def exact_match_scorer(output, expected, input):
        return {"exact_match": 1.0 if output == expected else 0.0}
    
    # Run an evaluation
    eval_data = [
        {"input": "Hello", "expected": "Hi there!"},
        {"input": "Goodbye", "expected": "See you later!"}
    ]
    
    result = bt.run_evaluation(
        "My AI App",
        "Test Evaluation", 
        eval_data,
        my_task,
        [exact_match_scorer]
    )
    
    print(f"Evaluation completed: {result.permalink}")
    print(f"Average scores: {result.scores}")
    
    # Use AI proxy
    proxy_response = bt.proxy_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        cache_mode=CacheMode.AUTO
    )
    
    # Log production event
    event_id = bt.log_production_event(
        "My AI App",
        input_data="User question",
        output_data="AI response",
        model="gpt-3.5-turbo",
        latency=0.5,
        tokens_used=150
    )
    
    print(f"Logged production event: {event_id}")


if __name__ == "__main__":
    example_usage()

