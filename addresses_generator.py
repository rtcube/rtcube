with open("addresses_all", "w+") as f:
    prefix = "192.168.143."
    ranges = [range(190, 206), (210, 226)]
    for r in ranges:
        for i in r:
            f.write(prefix + str(i) + "\n")
    #for i in range(190, 206):
    #    f.write(prefix + str(i) + "\n")
    #for i in range(210, 226):
    #    f.write(prefix + str(i) + "\n")

