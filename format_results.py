import os
import re
import argparse
from datetime import datetime

def parse_adaprompt_results(output_dir, task_names):
    """Parse AdaPromptCL results and format like clean-sprompts"""
    
    # Look for the latest results file
    result_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.txt') and 'log' in file:
            result_files.append(os.path.join(output_dir, file))
    
    if not result_files:
        print("No result files found in output directory")
        return
    
    # Use the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    
    # Parse results
    task_accuracies = {}
    final_avg = 0.0
    training_time = 0.0
    
    with open(latest_file, 'r') as f:
        content = f.read()
        
        # Extract task accuracies
        for i, task_name in enumerate(task_names):
            # Look for patterns like "Task 0 accuracy: 85.50%"
            pattern = f"Task {i}.*?accuracy.*?([0-9]+\.[0-9]+)"
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                task_accuracies[task_name] = float(matches[-1])  # Take the last match
        
        # Extract final average accuracy
        avg_pattern = r"Final.*?average.*?accuracy.*?([0-9]+\.[0-9]+)"
        avg_matches = re.findall(avg_pattern, content, re.IGNORECASE)
        if avg_matches:
            final_avg = float(avg_matches[-1])
        
        # Extract training time
        time_pattern = r"Total.*?time.*?([0-9]+\.[0-9]+)"
        time_matches = re.findall(time_pattern, content, re.IGNORECASE)
        if time_matches:
            training_time = float(time_matches[-1])
    
    return task_accuracies, final_avg, training_time

def format_results_like_clean_sprompts(task_accuracies, final_avg, training_time, task_names, output_file):
    """Format results in clean-sprompts style"""
    
    with open(output_file, 'w') as f:
        # Header
        header = "Tasks | " + " | ".join(task_names)
        f.write(header + "\n")
        
        # Results for each task
        for i, task_name in enumerate(task_names):
            row = f"task{i+1}"
            for eval_task in task_names:
                if eval_task in task_accuracies:
                    accuracy = task_accuracies[eval_task]
                    row += f" | {accuracy:.2f}%"
                else:
                    row += " | N/A"
            f.write(row + "\n")
        
        f.write("\n")
        f.write(f"Final Average Accuracy: {final_avg:.2f}%\n")
        f.write("\n")
        f.write(f"Total training time: {training_time:.2f}s\n")

def main():
    parser = argparse.ArgumentParser(description='Format AdaPromptCL results like clean-sprompts')
    parser.add_argument('--output_dir', type=str, required=True, help='AdaPromptCL output directory')
    parser.add_argument('--task_names', nargs='+', required=True, help='Task names')
    parser.add_argument('--result_file', type=str, help='Output result file name')
    
    args = parser.parse_args()
    
    if not args.result_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.result_file = f"deepfake_adaprompt_results_{timestamp}.txt"
    
    # Parse and format results
    task_accuracies, final_avg, training_time = parse_adaprompt_results(args.output_dir, args.task_names)
    
    # Save formatted results
    output_path = os.path.join(args.output_dir, args.result_file)
    format_results_like_clean_sprompts(task_accuracies, final_avg, training_time, args.task_names, output_path)
    
    print(f"Results formatted and saved to: {output_path}")
    print(f"Final Average Accuracy: {final_avg:.2f}%")

if __name__ == "__main__":
    main() 