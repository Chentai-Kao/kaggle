#!/usr/bin/python

import fann2
import fann2.libfann as libfann
import os

cities = {}
cityGroups = {}
restaurantTypes = {}

def collect(content, collector):
    if content not in collector:
        collector[content] = str(len(collector))

def parse(fileName):
    data = {}
    out = ""
    with open(fileName) as f:
        next(f)
        for line in f:
            fields = line.rstrip("\n").replace("/", ",").split(",")
            dataId = fields[0]
            city, cityGroup, restaurantType = tuple(fields[4:7])
            collect(city, cities)
            collect(cityGroup, cityGroups)
            collect(restaurantType, restaurantTypes)
            fields[4:7] = [cities[city], cityGroups[cityGroup],
                           restaurantTypes[restaurantType]]
            #for i in xrange(len(fields)):
            #    fields[i] = str(int(float(fields[i])))
            data[dataId] = fields[1:]

    outFile = os.path.splitext(fileName)[0] + ".data"
    with open(outFile, "w") as f:
        f.write("%d %d %d\n" % (len(data), len(data.values()[0]) - 1, 1))
        for d in data.values():
            f.write("%s\n" % " ".join(d[:-1]))
            f.write("%f\n" % (float(d[-1]) / 10000000))

    return data

def run_fann(dataFile):
    with open(dataFile) as f:
        connection_rate = 1
        learning_rate = 0.7
        num_hidden = 10
        _, num_input, num_output = tuple(map(lambda d: int(d),
                                             f.readline().split()))

        desired_error = 0.0001
        max_iterations = 10000
        iterations_between_reports = 1000

        ann = libfann.neural_net()
        ann.create_sparse_array(connection_rate,
                                (num_input, num_hidden, num_output))
        ann.set_learning_rate(learning_rate)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

        ann.train_on_file(dataFile, max_iterations,
                          iterations_between_reports, desired_error)

        ann.save(os.path.splitext(dataFile)[0] + ".net")

def main():
    train = parse("data/train.csv")
    run_fann("data/train.data")
    #run_fann("data/xor.data")

if __name__ == "__main__":
    main()
