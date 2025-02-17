import sys

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)

    file_name = sys.argv[1]
    print(f"Dataset file: {file_name}")

if __name__ == "__main__":
    describe()