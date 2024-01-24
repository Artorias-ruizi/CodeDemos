import sys
import time

from tqdm import tqdm


# 发呆0.5s
def action():
    time.sleep(0.5)


if __name__ == '__main__':

    with tqdm(range(10), total=10, leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for i in pbar:
            pbar.set_description('Epoch %d' % i)
            # 发呆0.5秒
            action()
            # 更新发呆进度
            pbar.update(1)
