from .args import get_args
from . import train
from . import test
from . import generate
from . import run

def main():

    args = get_args()

    if args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)
    elif args.mode == 'generate':
        generate.node_worker(args)
    elif args.mode == 'run':
        run.node_worker(args)

if __name__ == '__main__':
    main()
