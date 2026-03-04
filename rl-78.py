stores = [100, 80, 60]

for i in range(10):
    stores = [s - 10 if s > 20 else s + 50 for s in stores]
    print("Store levels:", stores)
