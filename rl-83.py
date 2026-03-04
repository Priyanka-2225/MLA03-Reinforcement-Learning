tasks = ["Pick", "Move", "Place"]

for t in tasks:
    print("High-Level Task:", t)
    for step in range(3):
        print("  Low-Level Action:", step)
