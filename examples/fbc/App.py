from multiprocessing.pool import ThreadPool

thread_pool = ThreadPool(processes=4)


def in_parallel(*functions):
    global thread_pool
    return thread_pool.map(lambda i: functions[i](), range(len(functions)), chunksize=1)

