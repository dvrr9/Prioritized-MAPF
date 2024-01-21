import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("name", help="Имя пользователя")
parser.add_argument("surname", help="Имя пользователя")
 
args = parser.parse_args()
 
print(f"Привет, {args.name}!")