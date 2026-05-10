#!/usr/bin/env python3
"""
Phase 6 Test Suite: Validation and Hardening

Test scenarios for the conversational agent:
1. Basic 2-turn conversation
2. Medium 10-turn conversation
3. Long 20-turn conversation forcing compaction
4. Tool success/failure scenarios
5. Regression checks

Run with: python3 test_phase6.py [scenario_name]
"""

import subprocess
import sys
from typing import Optional

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_agent_test(name: str, inputs: list[str], expected_patterns: Optional[list[str]] = None, timeout: int = 180) -> bool:
    """
    Run the agent with given inputs and check for expected patterns.
    
    Args:
        name: Test name
        inputs: List of user inputs (last should be 'q' to quit)
        expected_patterns: List of strings that should appear in output
        timeout: Timeout in seconds (default 180 for long tests)
    
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{BLUE}▶ Running: {name}{RESET}")
    
    # Prepare input string
    input_str = '\n'.join(inputs) + '\n'
    
    try:
        result = subprocess.run(
            ['./your_program.sh'],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout + result.stderr
        
        # Check for expected patterns
        if expected_patterns:
            for pattern in expected_patterns:
                if pattern not in output:
                    print(f"{RED}✗ FAILED: Missing expected pattern: '{pattern}'{RESET}")
                    print(f"Output preview:\n{output[:500]}")
                    return False
        
        # Basic sanity checks
        if "Budget:" not in output:
            print(f"{RED}✗ FAILED: No budget tracking in output{RESET}")
            return False
        
        if "usage_turn_summary" not in output and "usage_session_summary" not in output:
            print(f"{RED}✗ FAILED: No usage summary in output{RESET}")
            return False
        
        print(f"{GREEN}✓ PASSED{RESET}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"{RED}✗ FAILED: Test timed out (>{timeout}s){RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗ FAILED: {e}{RESET}")
        return False


def main():
    """Run Phase 6 test suite."""
    
    print(f"{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}Phase 6: Validation and Hardening - Test Suite{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    results = {}
    
    # Test 1: 2-turn conversation
    results["2-turn"] = run_agent_test(
        "2-turn Conversation",
        [
            "hello, what is an API?",
            "q"
        ],
        expected_patterns=[
            "API",
            "Budget:",
            "usage_session_summary"
        ]
    )
    
    # Test 2: 10-turn conversation (different skills)
    results["10-turn"] = run_agent_test(
        "10-turn Conversation (skill variety)",
        [
            "explain what machine learning is",
            "write a function to calculate fibonacci",
            "what are neural networks?",
            "debug this: my code crashes on None input",
            "explain backpropagation",
            "write a simple decorator function",
            "what is gradient descent?",
            "create a class for a queue",
            "explain overfitting",
            "q"
        ],
        expected_patterns=[
            "Budget:",
            "usage_session_summary",
            "def ",  # Should have code examples
            "Skill:"  # Should detect skills
        ]
    )
    
    # Test 3: Long conversation forcing compaction
    results["20-turn-compaction"] = run_agent_test(
        "20-turn Conversation (triggers compaction)",
        [
            "What is a token?",
            "Explain transformers",
            "What are attention mechanisms?",
            "How does BERT work?",
            "Explain positional encoding",
            "What are embeddings?",
            "Explain the attention equation",
            "What is self-attention?",
            "How is softmax used in attention?",
            "What is multi-head attention?",
            "Explain query, key, value matrices",
            "How does encoder work?",
            "Explain decoder in transformers",
            "What is cross-attention?",
            "How is gradient descent used?",
            "Explain backpropagation",
            "What are batch normalization benefits?",
            "Explain layer normalization",
            "What is residual connection?",
            "q"
        ],
        expected_patterns=[
            "Budget:",
            "usage_session_summary"
        ],
        timeout=240  # 20 turns might take longer
    )
    
    # Test 4: Tool usage (read files)
    results["tool-read"] = run_agent_test(
        "Tool Test: Read file",
        [
            "read the file app/main.py and show me the first 10 lines",
            "q"
        ],
        expected_patterns=[
            "Budget:",
            "app/main.py",
            "import",
            "usage_session_summary"
        ]
    )
    
    # Test 5: Default mode (no skill match)
    results["default-mode"] = run_agent_test(
        "Default Mode (no skill triggered)",
        [
            "what's the weather like today?",
            "tell me a joke",
            "q"
        ],
        expected_patterns=[
            "Budget:",
            "usage_session_summary"
        ]
    )
    
    # Print summary
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}Test Results Summary{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = f"{GREEN}✓ PASSED{RESET}" if passed_flag else f"{RED}✗ FAILED{RESET}"
        print(f"  {test_name:30} {status}")
    
    print(f"\n{YELLOW}Overall: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"{GREEN}All tests passed! ✓{RESET}")
        return 0
    else:
        print(f"{RED}Some tests failed. Review output above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
