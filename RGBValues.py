for red in range(0,256, 5):
    for green in range(0,256, 5):
        for blue in range(0, 256, 5):
            if (red*0.299 + green*0.587 + blue*0.114) > 186:
                print("{}, {}, {}, {}".format(red, green, blue, "0"), file=open("data/data.txt", "a"))
            else:
                print("{}, {}, {}, {}".format(red, green, blue, "1"), file=open("data/data.txt", "a"))