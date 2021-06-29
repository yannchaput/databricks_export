import argparse
parser = argparse.ArgumentParser()
parser.add_argument("alpha")
parser.add_argument("l1_ratio")
args = parser.parse_args()
print(args.alpha)
