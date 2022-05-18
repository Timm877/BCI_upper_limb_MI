import argparse
import src.utils_finetune_ft as utils_finetune

def execution(type):
    print(f'Initializing for Finetune Transfer learning...')
    utils_finetune.run()
    print('Finished')

def main():
    for type in FLAGS.type:
        print(type)
        execution(type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--type", nargs='+', default=['multiclass'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'multiclass'.")
    FLAGS, unparsed = parser.parse_known_args()
    main()