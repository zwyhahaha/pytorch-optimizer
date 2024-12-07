import subprocess


# commands = [
#     "CUDA_VISIBLE_DEVICES=7 python examples/mnist_os_test.py --lr 0.01 --beta 0.99 --beta_lr 0",
#     "CUDA_VISIBLE_DEVICES=1 python examples/mnist_os_test.py --lr 0.01 --beta 0.995 --beta_lr 0",
#     "CUDA_VISIBLE_DEVICES=2 python examples/mnist_test.py --lr 0.01 --beta 0.99 --beta_lr 0",
# ]

commands = [
    # "CUDA_VISIBLE_DEVICES=7 python examples/mnist_os_test.py --lr 0.01 --batch_size 6000 --optimizer_name NAG",
    # "CUDA_VISIBLE_DEVICES=1 python examples/mnist_os_test.py --lr 0.01 --batch_size 6000 --optimizer_name Adam",
    "CUDA_VISIBLE_DEVICES=2 python examples/mnist_os_test.py --lr 0.1 --batch_size 6000 --optimizer_name OSGM",
    "CUDA_VISIBLE_DEVICES=3 python examples/mnist_os_test.py --lr 0.1 --batch_size 6000 --optimizer_name OSMM",
    # "CUDA_VISIBLE_DEVICES=6 python examples/mnist_os_test.py --lr 0.03 --batch_size 6000 --optimizer_name OSGM",
    # "CUDA_VISIBLE_DEVICES=4 python examples/mnist_os_test.py --lr 0.4 --batch_size 6000 --optimizer_name OSMM",
]

# commands = [
#     "CUDA_VISIBLE_DEVICES=2 python cifar10/main.py --lr 0.01 --optimizer_name OSMM --beta 0.995 --beta_lr 1e-3",
#     # "CUDA_VISIBLE_DEVICES=3 python cifar10/main.py --lr 0.001 --optimizer_name OSMM --beta 0.1 --beta_lr 1e-2",
# ]

processes = []

for cmd in commands:
    print(f"Starting: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append(proc)

for i, proc in enumerate(processes):
    stdout, stderr = proc.communicate()
    if proc.returncode == 0:
        print(f"Command {i+1} finished successfully:\n{stdout.decode()}")
    else:
        print(f"Command {i+1} failed:\n{stderr.decode()}")
