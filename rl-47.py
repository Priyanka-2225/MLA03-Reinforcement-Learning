import threading

def worker():
    # local actor-critic training
    pass

for i in range(4):
    t=threading.Thread(target=worker)
    t.start()
