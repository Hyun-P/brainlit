base_path = "/data/jacsstorage/samples/tathey/bil1/"

for a in range(1,9):
    for b in range(1,9):
        for c in range(1,9):
            for d in range(1,9):
                for e in range(1,9):
                    for f in range(1,9):
                        path = base_path + f"/{a}/{d}/{c}/{d}/{e}/{f}/"
                        im = np.zeros(())
                        for slice in range