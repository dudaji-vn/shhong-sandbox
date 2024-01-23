from multiprocessing import Pool
import pytorch_client as pc
import triton_client as tc
import flask_client as fc
import time


#filename = 'small_img1.jpg'
filename = 'img1.jpg'
pool_size = 8
load_count = 10


print(f'{filename}, {load_count}')


def pool_run(pool_size, load_count, func):
    with Pool(pool_size) as p:
        t1 = time.time()
        p.map(func, [1] * load_count)
        t2 = time.time()
        diff = t2 - t1
        fps = load_count / diff
        return (diff, fps)
    return -1


def _run_flask(try_count):
    fc.run(filename)


def _run_triton(try_count):
    tc.run(filename)


def main_flask():
    print('load on argos-server')
    diff, fps = pool_run(pool_size, load_count, _run_flask)
    flask_fps = fps
    print(f'{load_count}, diff={diff}, fps={fps}')


def main_triton():
    print('load on triton')
    diff, fps = pool_run(pool_size, load_count, _run_triton)
    triton_fps = fps
    print(f'{load_count}, diff={diff}, fps={fps}')


def main_pytorch():
    model = pc.get_model()
    print('load on pytorch')
    t1 = time.time()
    for i in range(load_count):
        pc.run(model, filename)
    t2 = time.time()
    diff = t2 - t1
    fps = load_count / diff
    print(f'{load_count}, diff={diff}, fps={fps}')


main_triton()
main_pytorch()
main_flask()


