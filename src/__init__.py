import os


def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        "data",
        "models",
        "src"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create __init__.py files for src package
    init_files = [
        "src/__init__.py",
    ]

    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization file\n')
            print(f"Created: {init_file}")


if __name__ == "__main__":
    create_directory_structure()