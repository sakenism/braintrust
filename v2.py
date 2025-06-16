import os
import json
import csv
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import uuid

# Mock imports - replace with actual Braintrust imports
try:
    from braintrust import Eval, init, start_span, log
    from autoevals import Factuality, Levenshtein, BLEU, Security, Summarization
except ImportError:
    # Fallback for demo purposes
    print("Braintrust not installed. Using mock implementations.")

@dataclass
class EvalResult:
    """Result of an evaluation run"""
    experiment_id: str
    scores: Dict[str, float]
    total_cases: int
    passed_cases: int
    failed_cases: int
    metadata: Dict[str, Any]

@dataclass
class TraceSpan:
    """Represents a traced operation"""
    span_id: str
    name: str
    input_data: Any
    output_data: Any
    metadata: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None

class BraintrustSDK:
    """
    Simplified SDK class for Braintrust operations
    Handles evaluations, logging, tracing, data connections, and autoevals
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 project_name: str = "default-project",
                 base_url: str = "https://api.braintrust.dev"):
        """
        Initialize BraintrustSDK
        
        Args:
            api_key: Braintrust API key (defaults to BRAINTRUST_API_KEY env var)
            project_name: Name of the Braintrust project
            base_url: Braintrust API base URL
        """
        self.api_key = api_key or os.getenv("BRAINTRUST_API_KEY")
        self.project_name = project_name
        self.base_url = base_url
        self.active_spans = {}
        
        if not self.api_key:
            raise ValueError("API key required. Set BRAINTRUST_API_KEY environment variable or pass api_key parameter")
        
        # Initialize Braintrust logging
        self._init_braintrust()
    
    def _init_braintrust(self):
        """Initialize Braintrust connection"""
        try:
            # Initialize Braintrust with project
            init(project=self.project_name, api_key=self.api_key)
            print(f"✅ Braintrust initialized for project: {self.project_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Braintrust: {e}")
    
    # ========== EVALUATION TOOLS ==========
    
    def create_eval(self, 
                   name: str,
                   data: Union[List[Dict], Callable],
                   task: Callable,
                   scorers: List[str] = None,
                   custom_scorers: List[Callable] = None) -> EvalResult:
        """
        Create and run an evaluation
        
        Args:
            name: Name of the evaluation
            data: Test data (list of dicts or callable that returns data)
            task: Function to evaluate (takes input, returns output)
            scorers: List of built-in scorer names ['factuality', 'levenshtein', 'bleu', etc.]
            custom_scorers: List of custom scoring functions
            
        Returns:
            EvalResult object with evaluation results
        """
        print(f"🔍 Running evaluation: {name}")
        
        # Get data
        test_data = data() if callable(data) else data
        
        # Build scorer list
        scorer_functions = []
        if scorers:
            scorer_functions.extend(self._get_builtin_scorers(scorers))
        if custom_scorers:
            scorer_functions.extend(custom_scorers)
        
        if not scorer_functions:
            scorer_functions = [Levenshtein()]  # Default scorer
        
        try:
            # Run Braintrust evaluation
            experiment_id = str(uuid.uuid4())
            
            # Mock evaluation for demo - replace with actual Braintrust Eval
            result = self._run_evaluation(name, test_data, task, scorer_functions)
            
            print(f"✅ Evaluation completed: {result.scores}")
            return result
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise
    
    def _get_builtin_scorers(self, scorer_names: List[str]) -> List:
        """Get built-in autoevals scorers"""
        scorer_map = {
            'factuality': Factuality(),
            'levenshtein': Levenshtein(),
            'bleu': BLEU(),
            'security': Security(),
            'summarization': Summarization(),
        }
        
        return [scorer_map[name] for name in scorer_names if name in scorer_map]
    
    def _run_evaluation(self, name: str, data: List[Dict], task: Callable, scorers: List) -> EvalResult:
        """Internal method to run evaluation"""
        scores = {}
        passed = 0
        failed = 0
        
        for item in data:
            try:
                # Run task
                output = task(item.get('input'))
                
                # Calculate scores
                for scorer in scorers:
                    score_name = getattr(scorer, 'name', scorer.__class__.__name__)
                    if hasattr(scorer, '__call__'):
                        score = scorer(output, item.get('expected'), input=item.get('input'))
                        if hasattr(score, 'score'):
                            scores[score_name] = score.score
                        else:
                            scores[score_name] = score
                
                passed += 1
            except Exception as e:
                failed += 1
                print(f"⚠️  Test case failed: {e}")
        
        return EvalResult(
            experiment_id=str(uuid.uuid4()),
            scores=scores,
            total_cases=len(data),
            passed_cases=passed,
            failed_cases=failed,
            metadata={'name': name, 'timestamp': datetime.now().isoformat()}
        )
    
    # ========== LOGGING & TRACING ==========
    
    def start_trace(self, name: str, input_data: Any = None, metadata: Dict = None) -> str:
        """
        Start a new trace span
        
        Args:
            name: Name of the operation being traced
            input_data: Input data for the operation
            metadata: Additional metadata
            
        Returns:
            Span ID for the started trace
        """
        span_id = str(uuid.uuid4())
        span = TraceSpan(
            span_id=span_id,
            name=name,
            input_data=input_data,
            output_data=None,
            metadata=metadata or {},
            start_time=datetime.now()
        )
        
        self.active_spans[span_id] = span
        
        try:
            # Start Braintrust span
            braintrust_span = start_span(name=name)
            if input_data:
                braintrust_span.log(input=input_data)
        except:
            pass  # Continue even if Braintrust logging fails
        
        print(f"📊 Started trace: {name} (ID: {span_id[:8]})")
        return span_id
    
    def end_trace(self, span_id: str, output_data: Any = None, metadata: Dict = None):
        """
        End a trace span
        
        Args:
            span_id: ID of the span to end
            output_data: Output data from the operation
            metadata: Additional metadata
        """
        if span_id not in self.active_spans:
            print(f"⚠️  Warning: Span {span_id} not found")
            return
        
        span = self.active_spans[span_id]
        span.end_time = datetime.now()
        span.output_data = output_data
        if metadata:
            span.metadata.update(metadata)
        
        duration = (span.end_time - span.start_time).total_seconds()
        
        try:
            # Log to Braintrust
            log(
                input=span.input_data,
                output=output_data,
                metadata={**span.metadata, 'duration_seconds': duration}
            )
        except:
            pass  # Continue even if Braintrust logging fails
        
        print(f"📊 Ended trace: {span.name} (Duration: {duration:.2f}s)")
        del self.active_spans[span_id]
    
    def log_interaction(self, 
                       input_data: Any, 
                       output_data: Any, 
                       metadata: Dict = None,
                       experiment_name: str = None):
        """
        Log a single AI interaction
        
        Args:
            input_data: Input to the AI system
            output_data: Output from the AI system
            metadata: Additional metadata (model, temperature, etc.)
            experiment_name: Optional experiment name
        """
        try:
            log_data = {
                'input': input_data,
                'output': output_data,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            if experiment_name:
                log_data['experiment'] = experiment_name
            
            # Log to Braintrust
            log(**log_data)
            print(f"📝 Logged interaction: {str(input_data)[:50]}...")
            
        except Exception as e:
            print(f"❌ Failed to log interaction: {e}")
    
    # ========== DATABASE CONNECTIONS ==========
    
    def connect_sqlite(self, db_path: str) -> sqlite3.Connection:
        """
        Connect to SQLite database
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(db_path)
            print(f"📊 Connected to SQLite: {db_path}")
            return conn
        except Exception as e:
            print(f"❌ Failed to connect to SQLite: {e}")
            raise
    
    def query_database(self, 
                      connection: sqlite3.Connection, 
                      query: str, 
                      params: tuple = None) -> List[Dict]:
        """
        Execute database query and return results
        
        Args:
            connection: Database connection
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries representing query results
        """
        try:
            cursor = connection.cursor()
            cursor.execute(query, params or ())
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = [dict(zip(columns, row)) for row in rows]
            
            print(f"📊 Query executed: {len(results)} rows returned")
            return results
            
        except Exception as e:
            print(f"❌ Database query failed: {e}")
            raise
    
    def load_eval_data_from_db(self, 
                              connection: sqlite3.Connection, 
                              query: str,
                              input_col: str = 'input',
                              expected_col: str = 'expected') -> List[Dict]:
        """
        Load evaluation data from database
        
        Args:
            connection: Database connection
            query: SQL query to fetch test data
            input_col: Name of input column
            expected_col: Name of expected output column
            
        Returns:
            List of evaluation test cases
        """
        results = self.query_database(connection, query)
        
        eval_data = []
        for row in results:
            eval_data.append({
                'input': row.get(input_col),
                'expected': row.get(expected_col),
                'metadata': {k: v for k, v in row.items() 
                           if k not in [input_col, expected_col]}
            })
        
        print(f"📊 Loaded {len(eval_data)} test cases from database")
        return eval_data
    
    # ========== FILE OPERATIONS ==========
    
    def read_csv_file(self, file_path: str, **kwargs) -> List[Dict]:
        """
        Read CSV file and return as list of dictionaries
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            List of dictionaries representing CSV rows
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            data = df.to_dict('records')
            print(f"📄 Loaded CSV: {len(data)} rows from {file_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            raise
    
    def read_json_file(self, file_path: str) -> Union[Dict, List]:
        """
        Read JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📄 Loaded JSON: {file_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to read JSON: {e}")
            raise
    
    def load_eval_data_from_file(self, file_path: str, 
                                input_col: str = 'input',
                                expected_col: str = 'expected') -> List[Dict]:
        """
        Load evaluation data from file (CSV or JSON)
        
        Args:
            file_path: Path to data file
            input_col: Name of input column/field
            expected_col: Name of expected output column/field
            
        Returns:
            List of evaluation test cases
        """
        if file_path.endswith('.csv'):
            raw_data = self.read_csv_file(file_path)
        elif file_path.endswith('.json'):
            raw_data = self.read_json_file(file_path)
            if not isinstance(raw_data, list):
                raise ValueError("JSON file must contain an array of objects")
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        eval_data = []
        for row in raw_data:
            eval_data.append({
                'input': row.get(input_col),
                'expected': row.get(expected_col),
                'metadata': {k: v for k, v in row.items() 
                           if k not in [input_col, expected_col]}
            })
        
        print(f"📄 Loaded {len(eval_data)} test cases from {file_path}")
        return eval_data
    
    # ========== AUTOEVALS HELPERS ==========
    
    def available_scorers(self) -> List[str]:
        """Get list of available built-in scorers"""
        return ['factuality', 'levenshtein', 'bleu', 'security', 'summarization']
    
    def create_custom_scorer(self, name: str, scorer_func: Callable) -> Callable:
        """
        Create a custom scorer function
        
        Args:
            name: Name of the scorer
            scorer_func: Function that takes (output, expected, input) and returns score
            
        Returns:
            Scorer function with proper name attribute
        """
        def wrapped_scorer(output, expected=None, input=None):
            try:
                score = scorer_func(output, expected, input)
                return {
                    'name': name,
                    'score': score,
                    'metadata': {'custom_scorer': True}
                }
            except Exception as e:
                return {
                    'name': name,
                    'score': 0.0,
                    'metadata': {'error': str(e)}
                }
        
        wrapped_scorer.name = name
        return wrapped_scorer
    
    # ========== UTILITY METHODS ==========
    
    def create_eval_from_db(self, 
                           name: str,
                           db_path: str,
                           query: str,
                           task: Callable,
                           scorers: List[str] = None) -> EvalResult:
        """
        Convenience method to create evaluation from database
        
        Args:
            name: Evaluation name
            db_path: Path to SQLite database
            query: SQL query to fetch test data
            task: Function to evaluate
            scorers: List of scorer names
            
        Returns:
            EvalResult object
        """
        conn = self.connect_sqlite(db_path)
        try:
            data = self.load_eval_data_from_db(conn, query)
            return self.create_eval(name, data, task, scorers)
        finally:
            conn.close()
    
    def create_eval_from_file(self,
                             name: str,
                             file_path: str,
                             task: Callable,
                             scorers: List[str] = None,
                             input_col: str = 'input',
                             expected_col: str = 'expected') -> EvalResult:
        """
        Convenience method to create evaluation from file
        
        Args:
            name: Evaluation name
            file_path: Path to data file
            task: Function to evaluate
            scorers: List of scorer names
            input_col: Name of input column
            expected_col: Name of expected column
            
        Returns:
            EvalResult object
        """
        data = self.load_eval_data_from_file(file_path, input_col, expected_col)
        return self.create_eval(name, data, task, scorers)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up any remaining spans"""
        for span_id in list(self.active_spans.keys()):
            self.end_trace(span_id)


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Example usage of the BraintrustSDK
    
    # Initialize SDK
    sdk = BraintrustSDK(project_name="my-ai-project")
    
    # Example task function
    def simple_qa_task(input_text):
        """Simple QA task - replace with your AI model"""
        if "capital" in input_text.lower():
            return "Paris is the capital of France"
        return "I don't know"
    
    # Example 1: Create evaluation with inline data
    test_data = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is 2+2?", "expected": "4"},
    ]
    
    result = sdk.create_eval(
        name="Simple QA Test",
        data=test_data,
        task=simple_qa_task,
        scorers=['levenshtein', 'factuality']
    )
    
    # Example 2: Using tracing
    span_id = sdk.start_trace("AI Processing", input_data="Test input")
    output = simple_qa_task("What is the capital of France?")
    sdk.end_trace(span_id, output_data=output)
    
    # Example 3: Log interaction
    sdk.log_interaction(
        input_data="What is the capital of France?",
        output_data="Paris",
        metadata={"model": "gpt-4", "temperature": 0.7}
    )
    
    print("✅ BraintrustSDK examples completed!")

