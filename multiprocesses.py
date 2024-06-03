
import os
import time
import multiprocessing
from multiprocessing import Pool

CPU_COUNT = multiprocessing.cpu_count()


def func(idx):
    print(f'pid:{os.getpid()} \tidx:{idx}')
    time.sleep(1)
    return idx


if __name__ == "__main__":
    t1 = time.time()
    pool = Pool(CPU_COUNT)  # 创建拥有3个进程数量的进程池

    pool.map(func, range(80))
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    t2 = time.time()
    print("并行执行时间：", int(t2 - t1))
