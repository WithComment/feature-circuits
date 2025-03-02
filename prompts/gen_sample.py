import sys
import os
import random


def main():
  # Check if we have the correct number of arguments
  if len(sys.argv) != 3:
    print("Usage: python sample.py source-file target-dir")
    sys.exit(1)

  source_file = sys.argv[1]
  target_dir = sys.argv[2]

  # Check if source file exists
  if not os.path.isfile(source_file):
    print(f"Error: Source file '{source_file}' does not exist.")
    sys.exit(1)

  # Check if target directory exists, create if it doesn't
  if not os.path.exists(target_dir):
    try:
      os.makedirs(target_dir)
    except OSError:
      print(f"Error: Could not create target directory '{target_dir}'.")
      sys.exit(1)

  # Read all lines from the source file
  with open(source_file, 'r') as f:
    lines = f.readlines()

  # Shuffle the lines randomly
  random.shuffle(lines)

  # Split the lines into two halves
  middle = len(lines) // 2
  control_lines = lines[:middle]
  trial_lines = lines[middle:]

  # Write the first half to control.txt
  with open(os.path.join(target_dir, 'clean.txt'), 'w') as f:
    f.writelines(control_lines)

  # Write the second half to trial.txt
  with open(os.path.join(target_dir, 'patch.txt'), 'w') as f:
    f.writelines(trial_lines)

  print(f"Successfully split {len(lines)} lines:")
  print(
    f"  - {len(control_lines)} lines in {os.path.join(target_dir, 'clean.txt')}")
  print(f"  - {len(trial_lines)} lines in {os.path.join(target_dir, 'patch.txt')}")


if __name__ == "__main__":
  main()
