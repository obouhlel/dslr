import sys

# First of all, take a look at the available data. look in what format it is presented, if
# there are various types of data, the different ranges, and so on. It is important to make
# an idea of your raw material before starting. The more you work on data - the more you
# develop an intuition about how you will be able to use it.
# In this part, Professor McGonagall asks you to produce a program called describe.[extension].
# This program will take a dataset as a parameter.

def describe():
    if len(sys.argv) != 2:
        print("Usage: need to give a dataset")
        sys.exit(1)

    file_name = sys.argv[1]
    print(f"Dataset file: {file_name}")

if __name__ == "__main__":
    describe()