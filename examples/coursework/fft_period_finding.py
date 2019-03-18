import numpy


def gen_sequence(trash, period, size):
    per = [1e-4 * complex(numpy.random.random() - 0.5, numpy.random.random() - 0.5) for _ in range(period)]
    # per = [10 * (numpy.random.random() - 0.5) for _ in range(period)]
    gold = per * ((size - trash) // period + 1)
    trash_list = [1e-4 * complex(numpy.random.random() - 0.5, numpy.random.random() - 0.5) for _ in range(trash)]
    # trash_list = [10 * (numpy.random.random() - 0.5) for _ in range(trash)]
    return numpy.array(trash_list + gold[:size - trash], dtype=numpy.complex)
    # return numpy.array(trash_list + gold[:size - trash], dtype=numpy.float64)


def find_period_clean(seq):
    n = len(seq)
    w = numpy.absolute(numpy.fft.fft(seq))
    span = max(w) - min(w)

    period = 1
    k = 1
    for i in range(k, n - k):
        left, this, right = w[i - k], w[i], w[i + k]
        if left < this and right < this:
            period += 1

    return period


def find_period_noisy(seq):
    n = len(seq)
    w = numpy.absolute(numpy.fft.fft(seq))

    period = find_period_clean(seq)

    return range(n), w


def test_noisy_signal():
    sample_size = 64
    for p in range(1, sample_size // 4 + 1):
        res = find_period_clean(gen_sequence(0, p, sample_size))
        # print("{:3f}".format(res), p)
        if res != p:
            print(res, p)


# test_clean_signal()
test_noisy_signal()

# print("\n".join(map(str, gen_sequence(6, 2, 256))))


from matplotlib import pyplot as plt

plt.plot(*find_period_noisy(gen_sequence(16, 16, 64)))
plt.show()
